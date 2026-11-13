-- POST-only load: measures what durability costs on the submit path.
--   wrk -t4 -c64 -d10s --latency -s bench/wrk_submit.lua http://127.0.0.1:8080
wrk.method = "POST"
wrk.path = "/v1/jobs"
wrk.headers["Content-Type"] = "application/json"
wrk.body = '{"input": "chaos://ok"}'
