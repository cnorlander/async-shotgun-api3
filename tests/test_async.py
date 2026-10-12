"""Tests for the aiohttp-backed asynchronous client."""

import unittest

from aiohttp import web

import shotgun_api3


class AsyncShotgunTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []

        async def rpc(request):
            payload = await request.json()
            self.requests.append(payload)
            method = payload["method_name"]
            if method == "info":
                results = {"version": [9, 9, 9]}
            elif method == "read":
                results = {
                    "entities": [
                        {"type": "Shot", "id": 12, "code": "shot_010"}
                    ],
                    "paging_info": {"has_next_page": False},
                }
            elif method == "delete":
                results = True
            elif method in ("create", "update"):
                results = [{"type": "Shot", "id": 12}]
            elif method == "batch":
                results = [{"type": "Shot", "id": 12}]
            else:
                results = {"method": method}
            return web.json_response({"results": results})

        app = web.Application()
        app.router.add_post("/api3/json", rpc)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.sg = shotgun_api3.AsyncShotgun(
            "http://127.0.0.1:%s" % port,
            "test_script",
            "test_key",
        )

    async def asyncTearDown(self):
        await self.sg.close()
        await self.runner.cleanup()

    async def test_async_rpc_uses_aiohttp_and_authenticates(self):
        result = await self.sg.call_rpc("echo", {"hello": "world"})

        self.assertEqual({"method": "echo"}, result)
        self.assertEqual("info", self.requests[0]["method_name"])
        self.assertEqual("echo", self.requests[1]["method_name"])
        self.assertEqual(
            {
                "script_name": "test_script",
                "script_key": "test_key",
            },
            self.requests[1]["params"][0],
        )

    async def test_find_and_delete(self):
        async with self.sg:
            records = await self.sg.find("Shot", [], ["id", "code"])
            deleted = await self.sg.delete("Shot", 12)

        self.assertEqual(
            [{"type": "Shot", "id": 12, "code": "shot_010"}], records
        )
        self.assertTrue(deleted)

    async def test_create_update_and_batch(self):
        created = await self.sg.create("Shot", {"code": "shot_010"})
        updated = await self.sg.update("Shot", 12, {"code": "shot_020"})
        batched = await self.sg.batch(
            [{"request_type": "delete", "entity_type": "Shot", "entity_id": 12}]
        )

        self.assertEqual({"type": "Shot", "id": 12}, created)
        self.assertEqual({"type": "Shot", "id": 12}, updated)
        self.assertEqual([{"type": "Shot", "id": 12}], batched)
