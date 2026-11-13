-- wrk script: 0.2% POST /v1/jobs (an image), 99.8% GET /v1/jobs/{id} status polls.
-- Env: MHPREP_IMAGES=file with one image path per line, MHPREP_MAX_ID=highest job id.
local images = {}
for line in io.lines(os.getenv("MHPREP_IMAGES")) do images[#images + 1] = line end
local max_id = tonumber(os.getenv("MHPREP_MAX_ID") or "100")
local counter = 0

request = function()
  counter = counter + 1
  if counter % 500 == 0 then
    local body = '{"input":"' .. images[math.random(#images)] .. '","tenant":"wrk"}'
    return wrk.format("POST", "/v1/jobs", {["Content-Type"] = "application/json"}, body)
  end
  return wrk.format("GET", "/v1/jobs/" .. math.random(1, max_id) .. "?manifest=0")
end

done = function(summary, latency, requests)
  io.write(string.format('WRK_JSON {"requests":%d,"duration_us":%d,"errors":%d,' ..
    '"p50_us":%d,"p90_us":%d,"p99_us":%d,"p999_us":%d,"max_us":%d}\n',
    summary.requests, summary.duration,
    summary.errors.connect + summary.errors.read + summary.errors.write +
    summary.errors.status + summary.errors.timeout,
    latency:percentile(50), latency:percentile(90), latency:percentile(99),
    latency:percentile(99.9), latency.max))
end
