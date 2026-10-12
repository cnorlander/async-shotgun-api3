"""Asynchronous ShotGrid client backed by :mod:`aiohttp`."""

from __future__ import annotations

import asyncio
import json
import ssl
from typing import Any, Dict, List, Optional, Tuple, Union
from xmlrpc.client import ProtocolError

import aiohttp

from .shotgun import (
    ServerCapabilities,
    Shotgun,
    ShotgunError,
    _translate_filters,
)


class AsyncShotgun(Shotgun):
    """Async counterpart to :class:`Shotgun`, using a shared aiohttp session.

    Create a client and use it as an async context manager::

        async with AsyncShotgun(url, script_name, api_key) as sg:
            shots = await sg.find("Shot", filters, ["id", "code"])

    The HTTP session is created lazily on the first request and closed when the
    context exits. The synchronous :class:`Shotgun` client remains available.
    """

    def __init__(
        self,
        base_url: str,
        script_name: Optional[str] = None,
        api_key: Optional[str] = None,
        convert_datetimes_to_utc: bool = True,
        http_proxy: Optional[str] = None,
        connect: bool = False,
        ca_certs: Optional[str] = None,
        login: Optional[str] = None,
        password: Optional[str] = None,
        sudo_as_login: Optional[str] = None,
        session_token: Optional[str] = None,
        auth_token: Optional[str] = None,
    ) -> None:
        # Network I/O cannot happen in __init__; `connect` is retained for
        # constructor compatibility, and connection is established on demand.
        if connect:
            raise ValueError(
                "AsyncShotgun cannot connect in __init__; call await connect() instead."
            )
        if auth_token is not None and (login is None or password is None):
            raise ValueError(
                "must provide a user login and password with an auth_token"
            )
        if auth_token is not None and (script_name is not None or api_key is not None):
            raise ValueError("cannot provide an auth_code with script_name/api_key")
        super().__init__(
            base_url,
            script_name,
            api_key,
            convert_datetimes_to_utc,
            http_proxy,
            connect=False,
            ca_certs=ca_certs,
            login=login,
            password=password,
            sudo_as_login=sudo_as_login,
            session_token=session_token,
            auth_token=None,
        )
        self._async_auth_token = auth_token
        self.config.auth_token = auth_token
        self._aiohttp_ca_certs = self._get_certs_file(ca_certs)
        self._session: Optional[aiohttp.ClientSession] = None
        self._connect_lock = asyncio.Lock()
        self._connected = False

    def __getattribute__(self, name: str) -> Any:
        if (
            not name.startswith("_")
            and name in Shotgun.__dict__
            and callable(Shotgun.__dict__[name])
            and name not in type(self).__dict__
            and name not in {"add_user_agent", "reset_user_agent", "set_session_uuid"}
        ):
            raise NotImplementedError(
                "AsyncShotgun.%s is not implemented yet." % name
            )
        return super().__getattribute__(name)

    async def __aenter__(self) -> "AsyncShotgun":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.close()

    @property
    def server_caps(self) -> ServerCapabilities:
        if self._server_caps is None:
            raise ShotgunError("Await connect() before reading server capabilities.")
        return self._server_caps

    @property
    def server_info(self) -> Dict[str, Any]:
        return self.server_caps.server_info

    def _make_session(self) -> aiohttp.ClientSession:
        ssl_context = ssl.create_default_context(cafile=self._aiohttp_ca_certs)
        connector = aiohttp.TCPConnector(
            ssl=ssl_context,
            keepalive_timeout=self.config.max_connection_idle_secs or None,
        )
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_secs)
        return aiohttp.ClientSession(connector=connector, timeout=timeout)

    def _proxy(self) -> Tuple[Optional[str], Optional[aiohttp.BasicAuth]]:
        if not self.config.proxy_server:
            return None, None
        proxy = "http://%s:%s" % (
            self.config.proxy_server,
            self.config.proxy_port,
        )
        proxy_auth = None
        if self.config.proxy_user and self.config.proxy_pass:
            proxy_auth = aiohttp.BasicAuth(
                self.config.proxy_user, self.config.proxy_pass
            )
        return proxy, proxy_auth

    async def connect(self) -> None:
        """Establish the aiohttp session and load server capabilities."""
        async with self._connect_lock:
            if self._session is None or self._session.closed:
                self._session = self._make_session()
            if self._connected:
                return

            info = await self._rpc("info", None, include_auth_params=False)
            self._server_caps = ServerCapabilities(self.config.server, info)
            self._connected = True

            if self._async_auth_token:
                token_result = await self._rpc("get_session_token", None)
                session_token = (token_result or {}).get("session_id")
                if not session_token:
                    raise RuntimeError(
                        "Could not extract session_id from %s" % token_result
                    )
                self.config.session_token = session_token
                self.config.user_login = None
                self.config.user_password = None
                self._async_auth_token = None
                self.config.auth_token = None

    async def close(self) -> None:
        """Close the aiohttp session; a later request may reconnect."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
        self._connected = False

    async def _ensure_connected(self) -> None:
        if not self._connected:
            await self.connect()

    async def _rpc(
        self,
        method: str,
        params: Any,
        include_auth_params: bool = True,
        first: bool = False,
    ) -> Any:
        if include_auth_params:
            await self._ensure_connected()
        elif self._session is None or self._session.closed:
            self._session = self._make_session()

        payload = self._build_payload(
            method, self._transform_outbound(params), include_auth_params
        )
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Connection": "keep-alive",
            "User-Agent": "; ".join(self._user_agents),
        }
        if self.config.localized:
            headers["locale"] = "auto"
        if self.config.authorization:
            headers["Authorization"] = self.config.authorization

        url = "%s://%s%s" % (
            self.config.scheme,
            self.config.server,
            self.config.api_path,
        )
        proxy, proxy_auth = self._proxy()
        session = self._session
        if session is None:
            raise RuntimeError("HTTP session was not initialized.")

        for attempt in range(1, self.config.max_rpc_attempts + 1):
            try:
                async with session.post(
                    url,
                    data=body,
                    headers=headers,
                    proxy=proxy,
                    proxy_auth=proxy_auth,
                ) as response:
                    response_body = await response.text()
                    response_headers = {
                        key.lower(): value for key, value in response.headers.items()
                    }
                    status = (response.status, response.reason or "")
                    try:
                        self._parse_http_status(status)
                    except ProtocolError as error:
                        error.headers = response_headers
                        if attempt < self.MAX_ATTEMPTS and error.errcode in (502, 504):
                            await asyncio.sleep(attempt * self.BACKOFF)
                            continue
                        if error.errcode == 403:
                            error.errmsg += ": %s" % response_body
                        raise

                decoded = self._decode_response(response_headers, response_body)
                self._response_errors(decoded)
                decoded = self._transform_inbound(decoded)
                if not isinstance(decoded, dict) or "results" not in decoded:
                    return decoded
                results = decoded.get("results")
                if first and isinstance(results, list):
                    return results[0]
                return results
            except (aiohttp.ClientError, asyncio.TimeoutError, ssl.SSLError):
                if attempt >= self.config.max_rpc_attempts:
                    raise
                await asyncio.sleep(self.config.rpc_attempt_interval / 1000.0)

    async def call_rpc(self, method: str, params: Any = None) -> Any:
        """Call an API RPC method asynchronously.

        This low-level method also provides access to ShotGrid RPC methods that
        do not yet have a dedicated async convenience method.
        """
        return await self._rpc(method, params)

    async def info(self) -> Dict[str, Any]:
        """Fetch API metadata from the server."""
        return await self._rpc("info", None, include_auth_params=False)

    async def find_one(
        self,
        entity_type: str,
        filters: Union[List, Tuple, Dict[str, Any]],
        fields: Optional[List[str]] = None,
        order: Optional[List[Dict[str, Any]]] = None,
        filter_operator: Optional[str] = None,
        retired_only: bool = False,
        include_archived_projects: bool = True,
        additional_filter_presets: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        results = await self.find(
            entity_type,
            filters,
            fields,
            order,
            filter_operator,
            1,
            retired_only,
            include_archived_projects=include_archived_projects,
            additional_filter_presets=additional_filter_presets,
        )
        return results[0] if results else None

    async def find(
        self,
        entity_type: str,
        filters: Union[List, Tuple, Dict[str, Any]],
        fields: Optional[List[str]] = None,
        order: Optional[List[Dict[str, Any]]] = None,
        filter_operator: Optional[str] = None,
        limit: int = 0,
        retired_only: bool = False,
        page: int = 0,
        include_archived_projects: bool = True,
        additional_filter_presets: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        await self._ensure_connected()
        if not isinstance(limit, int) or limit < 0:
            raise ValueError("limit parameter must be a positive integer")
        if not isinstance(page, int) or page < 0:
            raise ValueError("page parameter must be a positive integer")
        if isinstance(filters, (list, tuple)):
            filters = _translate_filters(filters, filter_operator)
        elif filter_operator:
            raise ShotgunError(
                "Deprecated: Use of filter_operator for find() is not valid any more."
                " See the documentation on find()"
            )

        if not include_archived_projects:
            self.server_caps.ensure_include_archived_projects()
        if additional_filter_presets:
            self.server_caps.ensure_support_for_additional_filter_presets()

        params = self._construct_read_parameters(
            entity_type,
            fields,
            filters,
            retired_only,
            order,
            include_archived_projects,
            additional_filter_presets,
        )
        if self.server_caps.ensure_return_image_urls_support():
            params["api_return_image_urls"] = True
        paging_param = (
            "return_paging_info_without_counts"
            if self.server_caps.ensure_paging_info_without_counts_support()
            else "return_paging_info"
        )
        params[paging_param] = False

        if limit and limit <= self.config.records_per_page:
            params["paging"]["entities_per_page"] = limit
            if page == 0:
                page = 1
        if page:
            params["paging"]["current_page"] = page
            result = await self._rpc("read", params)
            return self._parse_records(result.get("entities", []))

        params[paging_param] = True
        records = []
        params["paging"]["current_page"] = 1
        while True:
            result = await self._rpc("read", params)
            records.extend(result.get("entities", []))
            if limit and len(records) >= limit:
                records = records[:limit]
                break
            if "has_next_page" in result.get("paging_info", {}):
                if not result["paging_info"]["has_next_page"]:
                    break
            elif len(records) == result["paging_info"]["entity_count"]:
                break
            params["paging"]["current_page"] += 1
        return self._parse_records(records)

    async def create(
        self,
        entity_type: str,
        data: Dict[str, Any],
        return_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if "filmstrip_image" in data:
            await self._ensure_connected()
            if self.server_caps.version < (3, 1, 0):
                raise ShotgunError(
                    "Filmstrip thumbnail support requires server version 3.1 or "
                    "higher, server is %s" % (self.server_caps.version,)
                )
        if data.get("image") or data.get("filmstrip_image"):
            raise NotImplementedError("Async file uploads are not implemented yet.")
        data = {
            key: value
            for key, value in data.items()
            if key not in ("image", "filmstrip_image")
        }
        params = {
            "type": entity_type,
            "fields": self._dict_to_list(data),
            "return_fields": return_fields or ["id"],
        }
        record = await self._rpc("create", params, first=True)
        return self._parse_records(record)[0]

    async def update(
        self,
        entity_type: str,
        entity_id: int,
        data: Dict[str, Any],
        multi_entity_update_modes: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        data = data.copy()
        if "filmstrip_image" in data:
            await self._ensure_connected()
            if self.server_caps.version < (3, 1, 0):
                raise ShotgunError(
                    "Filmstrip thumbnail support requires server version 3.1 or "
                    "higher, server is %s" % (self.server_caps.version,)
                )
        if (data.get("image") is not None and "image" in data) or data.get(
            "filmstrip_image"
        ):
            raise NotImplementedError("Async file uploads are not implemented yet.")
        if "filmstrip_image" in data:
            data.pop("filmstrip_image")
        if not data:
            return {"id": entity_id, "type": entity_type}
        params = self._translate_update_params(
            entity_type, entity_id, data, multi_entity_update_modes
        )
        record = await self._rpc("update", params)
        return self._parse_records(record)[0]

    async def delete(self, entity_type: str, entity_id: int) -> bool:
        return await self._rpc("delete", {"type": entity_type, "id": entity_id})

    async def revive(self, entity_type: str, entity_id: int) -> bool:
        return await self._rpc("revive", {"type": entity_type, "id": entity_id})

    async def batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(requests, list):
            raise ShotgunError(
                "batch() expects a list. Instead was sent a %s" % type(requests)
            )
        if not requests:
            return []

        calls = []
        for request in requests:
            request_type = request.get("request_type")
            entity_type = request.get("entity_type")
            if not request_type or not entity_type:
                raise ShotgunError(
                    "Batched request missing required key: request_type or entity_type."
                )
            params = {"request_type": request_type, "type": entity_type}
            if request_type == "create":
                if "data" not in request:
                    raise ShotgunError("Batched create request missing required key: data.")
                params["fields"] = self._dict_to_list(request["data"])
                params["return_fields"] = request.get("return_fields") or ["id"]
            elif request_type == "update":
                if "entity_id" not in request or "data" not in request:
                    raise ShotgunError(
                        "Batched update request missing required key: entity_id or data."
                    )
                params["id"] = request["entity_id"]
                params["fields"] = self._dict_to_list(
                    request["data"],
                    extra_data=self._dict_to_extra_data(
                        request.get("multi_entity_update_modes"),
                        "multi_entity_update_mode",
                    ),
                )
            elif request_type == "delete":
                if "entity_id" not in request:
                    raise ShotgunError(
                        "Batched delete request missing required key: entity_id."
                    )
                params["id"] = request["entity_id"]
            else:
                raise ShotgunError(
                    "Invalid request_type '%s' for batch" % request_type
                )
            calls.append(params)
        records = await self._rpc("batch", calls)
        return self._parse_records(records)

    async def summarize(
        self,
        entity_type: str,
        filters: Union[List, Dict[str, Any]],
        summary_fields: List[Dict[str, str]],
        filter_operator: Optional[str] = None,
        grouping: Optional[List[Dict[str, str]]] = None,
        include_archived_projects: bool = True,
    ) -> Dict[str, Any]:
        await self._ensure_connected()
        if not isinstance(grouping, list) and grouping is not None:
            raise ValueError("summarize() 'grouping' parameter must be a list or None")
        if isinstance(filters, (list, tuple)):
            filters = _translate_filters(filters, filter_operator)
        if not include_archived_projects:
            self.server_caps.ensure_include_archived_projects()
        params = {"type": entity_type, "summaries": summary_fields, "filters": filters}
        if include_archived_projects is False:
            params["include_archived_projects"] = False
        if grouping is not None:
            params["grouping"] = grouping
        return await self._rpc("summarize", params)
