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
                requested_type = payload["params"][-1]["type"]
                if requested_type == "HumanUser":
                    entity = {"type": "HumanUser", "id": 23, "login": "artist"}
                else:
                    entity = {"type": "Shot", "id": 12, "code": "shot_010"}
                results = {
                    "entities": [entity],
                    "paging_info": {"has_next_page": False},
                }
            elif method == "delete":
                results = True
            elif method in ("create", "update"):
                results = [{"type": "Shot", "id": 12}]
            elif method == "batch":
                results = [{"type": "Shot", "id": 12}]
            elif method == "get_session_token":
                results = {"session_id": "session-token"}
            elif method == "user_subscriptions_create":
                results = {"status": "success"}
            elif method == "schema_field_create":
                results = "sg_description"
            elif method == "query_display_name_cache":
                results = [{"matches": [], "terms": []}]
            elif method == "activity_stream":
                results = [{"updates": []}]
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

    async def test_remaining_rpc_methods(self):
        user = {"type": "HumanUser", "id": 23}
        entity = {"type": "Shot", "id": 12}

        self.assertEqual(
            {"method": "work_schedule_read"},
            await self.sg.work_schedule_read("2026-01-01", "2026-01-05"),
        )
        self.assertEqual(
            {"method": "work_schedule_update"},
            await self.sg.work_schedule_update("2026-01-01", True),
        )
        self.assertEqual(
            {"method": "export_page"}, await self.sg.export_page(12, "csv")
        )
        self.assertEqual(
            {"method": "follow"}, await self.sg.follow(user, entity)
        )
        self.assertEqual(
            {"method": "unfollow"}, await self.sg.unfollow(user, entity)
        )
        self.assertEqual(
            {"method": "followers"}, await self.sg.followers(entity)
        )
        self.assertEqual(
            {"method": "following"}, await self.sg.following(user)
        )
        self.assertEqual(
            {"method": "schema_entity_read"}, await self.sg.schema_entity_read()
        )
        self.assertEqual({"method": "schema_read"}, await self.sg.schema_read())
        self.assertEqual(
            {"method": "schema_field_read"},
            await self.sg.schema_field_read("Shot", "code"),
        )
        self.assertEqual(
            "sg_description",
            await self.sg.schema_field_create("Shot", "text", "Description"),
        )
        self.assertEqual(
            {"method": "schema_field_update"},
            await self.sg.schema_field_update("Shot", "code", {"name": "Code"}),
        )
        self.assertEqual(
            {"method": "schema_field_delete"},
            await self.sg.schema_field_delete("Shot", "code"),
        )
        self.assertEqual(
            {"type": "HumanUser", "id": 23, "login": "artist"},
            await self.sg.authenticate_human_user("artist", "password"),
        )
        self.assertEqual(
            {"script_name": "test_script", "script_key": "test_key"},
            self.sg._auth_params(),
        )
        auth_read = next(
            req
            for req in self.requests
            if req["method_name"] == "read"
            and req["params"][0].get("user_login") == "artist"
        )
        self.assertEqual("artist", auth_read["params"][0]["user_login"])
        self.assertIsNone(await self.sg.update_project_last_accessed(
            {"type": "Project", "id": 2}, user=user
        ))
        self.assertEqual(
            [{"method": "note_thread_contents"}],
            await self.sg.note_thread_read(12),
        )
        self.assertEqual(
            {"matches": [], "terms": []},
            await self.sg.text_search("shots", {"Shot": []}),
        )
        self.assertEqual(
            {"updates": []},
            await self.sg.activity_stream_read("Shot", 12),
        )
        self.assertEqual({"method": "nav_expand"}, await self.sg.nav_expand("/"))
        self.assertEqual(
            {"method": "nav_search"},
            await self.sg.nav_search_string("/", "shot"),
        )
        self.assertEqual(
            {"method": "nav_search"},
            await self.sg.nav_search_entity("/", entity),
        )
        self.assertEqual("session-token", await self.sg.get_session_token())
        self.assertEqual(
            {"method": "preferences_read"}, await self.sg.preferences_read()
        )
        self.assertEqual(
            {"method": "user_subscriptions_read"},
            await self.sg.user_subscriptions_read(),
        )
        self.assertTrue(await self.sg.user_subscriptions_create([]))
