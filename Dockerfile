# Multi-stage build: compile + test, then a slim runtime image.
FROM ubuntu:24.04 AS build
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build pkg-config ca-certificates curl \
      libopencv-dev libavformat-dev libavcodec-dev libswscale-dev libavutil-dev \
      nlohmann-json3-dev libcpp-httplib-dev libgtest-dev libpq-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN scripts/fetch_models.sh \
    && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build \
    && (cd build && ctest --output-on-failure)

FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libopencv-core406t64 libopencv-imgproc406t64 libopencv-imgcodecs406t64 \
      libopencv-videoio406t64 libopencv-dnn406t64 libcpp-httplib0.14t64 \
      libavformat60 libavcodec60 libswscale7 libavutil58 libpq5 tini \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build/mhprep /usr/local/bin/mhprep
COPY --from=build /src/models /usr/local/share/mhprep/models
ENV MHPREP_MODELS=/usr/local/share/mhprep/models
EXPOSE 8080
# tini forwards SIGTERM so `docker stop` triggers a graceful drain.
ENTRYPOINT ["/usr/bin/tini", "--", "mhprep"]
CMD ["serve", "--port", "8080", "--out", "/data/out"]
