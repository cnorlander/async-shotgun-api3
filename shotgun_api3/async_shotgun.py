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
        auth_params_override: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if include_auth_params:
            await self._ensure_connected()
        elif self._session is None or self._session.closed:
            self._session = self._make_session()

        outbound_params = self._transform_outbound(params)
        if auth_params_override is None:
            payload = self._build_payload(
                method, outbound_params, include_auth_params
            )
        else:
            call_params = [auth_params_override]
            if outbound_params:
                call_params.append(outbound_params)
            payload = {"method_name": method, "params": call_params}
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
        _auth_params_override: Optional[Dict[str, Any]] = None,
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
            _auth_params_override=_auth_params_override,
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
        _auth_params_override: Optional[Dict[str, Any]] = None,
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
            result = await self._rpc(
                "read", params, auth_params_override=_auth_params_override
            )
            return self._parse_records(result.get("entities", []))

        params[paging_param] = True
        records = []
        params["paging"]["current_page"] = 1
        while True:
            result = await self._rpc(
                "read", params, auth_params_override=_auth_params_override
            )
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

    async def work_schedule_read(
        self,
        start_date: str,
        end_date: str,
        project: Optional[Dict[str, Any]] = None,
        user: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        await self._ensure_connected()
        if self.server_caps.version < (3, 2, 0):
            raise ShotgunError(
                "Work schedule support requires server version 3.2 or higher, "
                "server is %s" % (self.server_caps.version,)
            )
        if not isinstance(start_date, str) or not isinstance(end_date, str):
            raise ShotgunError(
                "The start_date and end_date arguments must be strings in YYYY-MM-DD format"
            )
        return await self._rpc(
            "work_schedule_read",
            dict(start_date=start_date, end_date=end_date, project=project, user=user),
        )

    async def work_schedule_update(
        self,
        date: str,
        working: bool,
        description: Optional[str] = None,
        project: Optional[Dict[str, Any]] = None,
        user: Optional[Dict[str, Any]] = None,
        recalculate_field: Optional[str] = None,
    ) -> Dict[str, Any]:
        await self._ensure_connected()
        if self.server_caps.version < (3, 2, 0):
            raise ShotgunError(
                "Work schedule support requires server version 3.2 or higher, "
                "server is %s" % (self.server_caps.version,)
            )
        if not isinstance(date, str):
            raise ShotgunError("The date argument must be string in YYYY-MM-DD format")
        return await self._rpc(
            "work_schedule_update",
            dict(
                date=date,
                working=working,
                description=description,
                project=project,
                user=user,
                recalculate_field=recalculate_field,
            ),
        )

    async def export_page(self, page_id: int, format: str, layout_name=None) -> str:
        return await self._rpc(
            "export_page",
            dict(format=format, page_id=page_id, layout_name=layout_name),
        )

    async def follow(
        self, user: Dict[str, Any], entity: Dict[str, Any]
    ) -> Dict[str, Any]:
        await self._ensure_connected()
        if self.server_caps.version < (5, 1, 22):
            raise ShotgunError(
                "Follow support requires server version 5.2 or higher, "
                "server is %s" % (self.server_caps.version,)
            )
        return await self._rpc("follow", dict(user=user, entity=entity))

    async def unfollow(
        self, user: Dict[str, Any], entity: Dict[str, Any]
    ) -> Dict[str, Any]:
        await self._ensure_connected()
        if self.server_caps.version < (5, 1, 22):
            raise ShotgunError(
                "Follow support requires server version 5.2 or higher, "
                "server is %s" % (self.server_caps.version,)
            )
        return await self._rpc("unfollow", dict(user=user, entity=entity))

    async def followers(self, entity: Dict[str, Any]) -> List[Dict[str, Any]]:
        await self._ensure_connected()
        if self.server_caps.version < (5, 1, 22):
            raise ShotgunError(
                "Follow support requires server version 5.2 or higher, "
                "server is %s" % (self.server_caps.version,)
            )
        return await self._rpc("followers", {"entity": entity})

    async def following(
        self,
        user: Dict[str, Any],
        project: Optional[Dict[str, Any]] = None,
        entity_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        await self._ensure_connected()
        self.server_caps.ensure_user_following_support()
        params = {"user": user}
        if project:
            params["project"] = project
        if entity_type:
            params["entity_type"] = entity_type
        return await self._rpc("following", params)

    async def schema_entity_read(
        self, project_entity: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        await self._ensure_connected()
        params = self._add_project_param({}, project_entity)
        return await self._rpc("schema_entity_read", params or None)

    async def schema_read(
        self, project_entity: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        await self._ensure_connected()
        params = self._add_project_param({}, project_entity)
        return await self._rpc("schema_read", params or None)

    async def schema_field_read(
        self,
        entity_type: str,
        field_name: Optional[str] = None,
        project_entity: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        await self._ensure_connected()
        params = {"type": entity_type}
        if field_name:
            params["field_name"] = field_name
        params = self._add_project_param(params, project_entity)
        return await self._rpc("schema_field_read", params)

    async def schema_field_create(
        self,
        entity_type: str,
        data_type: str,
        display_name: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        params = {
            "type": entity_type,
            "data_type": data_type,
            "properties": [{"property_name": "name", "value": display_name}],
        }
        params["properties"].extend(
            self._dict_to_list(
                properties, key_name="property_name", value_name="value"
            )
        )
        return await self._rpc("schema_field_create", params)

    async def schema_field_update(
        self,
        entity_type: str,
        field_name: str,
        properties: Dict[str, Any],
        project_entity: Optional[Dict[str, Any]] = None,
    ) -> bool:
        params = {
            "type": entity_type,
            "field_name": field_name,
            "properties": [
                {"property_name": key, "value": value}
                for key, value in (properties or {}).items()
            ],
        }
        params = self._add_project_param(params, project_entity)
        return await self._rpc("schema_field_update", params)

    async def schema_field_delete(self, entity_type: str, field_name: str) -> bool:
        return await self._rpc(
            "schema_field_delete", {"type": entity_type, "field_name": field_name}
        )

    async def authenticate_human_user(
        self, user_login: str, user_password: str, auth_token: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        if not user_login:
            raise ValueError("Please supply a username to authenticate.")
        if not user_password:
            raise ValueError("Please supply a password for the user.")
        auth = {"user_login": str(user_login), "user_password": str(user_password)}
        if auth_token:
            auth["auth_token"] = str(auth_token)
        return await self.find_one(
            "HumanUser",
            [["sg_status_list", "is", "act"], ["login", "is", user_login]],
            ["id", "login"],
            "",
            "all",
            _auth_params_override=auth,
        )

    async def update_project_last_accessed(
        self,
        project: Dict[str, Any],
        user: Optional[Dict[str, Any]] = None,
    ) -> None:
        await self._ensure_connected()
        if self.server_caps.version < (5, 3, 20):
            raise ShotgunError(
                "update_project_last_accessed requires server version 5.3.20 or "
                "higher, server is %s" % (self.server_caps.version,)
            )
        if not user:
            login = self.config.sudo_as_login or self.config.user_login
            if login:
                user = await self.find_one("HumanUser", [["login", "is", login]])
        params = {"project_id": project["id"]}
        if user:
            params["user_id"] = user["id"]
        result = await self._rpc(
            "update_project_last_accessed_by_current_user", params
        )
        self._parse_records(result)

    async def note_thread_read(
        self, note_id: int, entity_fields: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        await self._ensure_connected()
        if self.server_caps.version < (6, 2, 0):
            raise ShotgunError(
                "note_thread requires server version 6.2.0 or higher, "
                "server is %s" % (self.server_caps.version,)
            )
        entity_fields = entity_fields or {}
        if not isinstance(entity_fields, dict):
            raise ValueError("entity_fields parameter must be a dictionary")
        record = await self._rpc(
            "note_thread_contents",
            {"note_id": note_id, "entity_fields": entity_fields},
        )
        return self._parse_records(record)

    async def text_search(
        self,
        text: str,
        entity_types: Dict[str, Any],
        project_ids: Optional[List] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        await self._ensure_connected()
        if self.server_caps.version < (6, 2, 0):
            raise ShotgunError(
                "auto_complete requires server version 6.2.0 or higher, "
                "server is %s" % (self.server_caps.version,)
            )
        if not isinstance(entity_types, dict):
            raise ValueError("entity_types parameter must be a dictionary")
        api_entity_types = {}
        for entity_type, filter_list in entity_types.items():
            if not isinstance(filter_list, (list, tuple)):
                raise ValueError(
                    "value of entity_types['%s'] must be a list or tuple."
                    % entity_type
                )
            api_entity_types[entity_type] = _translate_filters(
                filter_list, filter_operator=None
            )
        record = await self._rpc(
            "query_display_name_cache",
            {
                "text": text,
                "entity_types": api_entity_types,
                "project_ids": project_ids or [],
                "max_results": limit,
            },
        )
        return self._parse_records(record)[0]

    async def activity_stream_read(
        self,
        entity_type: str,
        entity_id: int,
        entity_fields: Optional[Dict[str, Any]] = None,
        min_id: Optional[int] = None,
        max_id: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        await self._ensure_connected()
        if self.server_caps.version < (6, 2, 0):
            raise ShotgunError(
                "activity_stream requires server version 6.2.0 or higher, "
                "server is %s" % (self.server_caps.version,)
            )
        entity_fields = entity_fields or {}
        if not isinstance(entity_fields, dict):
            raise ValueError("entity_fields parameter must be a dictionary")
        record = await self._rpc(
            "activity_stream",
            {
                "type": entity_type,
                "id": entity_id,
                "max_id": max_id,
                "min_id": min_id,
                "limit": limit,
                "entity_fields": entity_fields,
            },
        )
        return self._parse_records(record)[0]

    async def nav_expand(
        self, path: str, seed_entity_field=None, entity_fields=None
    ) -> Any:
        return await self._rpc(
            "nav_expand",
            {
                "path": path,
                "seed_entity_field": seed_entity_field,
                "entity_fields": entity_fields,
            },
        )

    async def nav_search_string(
        self, root_path: str, search_string: str, seed_entity_field=None
    ) -> Any:
        return await self._rpc(
            "nav_search",
            {
                "root_path": root_path,
                "seed_entity_field": seed_entity_field,
                "search_criteria": {"search_string": search_string},
            },
        )

    async def nav_search_entity(
        self,
        root_path: str,
        entity: Dict[str, Any],
        seed_entity_field: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return await self._rpc(
            "nav_search",
            {
                "root_path": root_path,
                "seed_entity_field": seed_entity_field,
                "search_criteria": {"entity": entity},
            },
        )

    async def get_session_token(self) -> str:
        if self.config.session_token:
            return self.config.session_token
        result = await self._rpc("get_session_token", None)
        session_token = (result or {}).get("session_id")
        if not session_token:
            raise RuntimeError("Could not extract session_id from %s" % result)
        self.config.session_token = session_token
        return session_token

    async def preferences_read(self, prefs: Optional[List] = None) -> Dict[str, Any]:
        await self._ensure_connected()
        if self.server_caps.version < (7, 10, 0):
            raise ShotgunError(
                "preferences_read requires server version 7.10.0 or higher, "
                "server is %s" % (self.server_caps.version,)
            )
        return await self._rpc("preferences_read", {"prefs": prefs or []})

    async def user_subscriptions_read(self) -> List:
        return await self._rpc("user_subscriptions_read", None)

    async def user_subscriptions_create(
        self, users: List[Dict[str, Union[str, List[str], None]]]
    ) -> bool:
        result = await self._rpc("user_subscriptions_create", {"users": users})
        return isinstance(result, dict) and result.get("status") == "success"

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
