# syntax=docker/dockerfile:1

# ------------------------------------------------------------------------------
# Stage 1: web frontend build
# vite is configured with outDir '../public', so the bundle lands in /app/public
# ------------------------------------------------------------------------------
FROM node:20-alpine AS web-build
WORKDIR /app/web

COPY web/package.json web/package-lock.json ./
RUN npm ci

COPY web/ ./
RUN npm run build

# ------------------------------------------------------------------------------
# Stage 2: C++ engine build (Release, -mavx2 -mfma -O3 set in CMakeLists.txt)
# ------------------------------------------------------------------------------
FROM ubuntu:24.04 AS engine-build
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        g++ cmake ninja-build git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY CMakeLists.txt ./
COPY include/ include/
COPY src/ src/
COPY examples/ examples/
COPY benchmarks/ benchmarks/
COPY tests/ tests/

RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="-mavx2 -mfma -O3" \
    && cmake --build build --target nexus_server --parallel

# ------------------------------------------------------------------------------
# Stage 3: minimal runtime
# ------------------------------------------------------------------------------
FROM ubuntu:24.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=engine-build /src/build/nexus_server ./nexus_server
COPY --from=web-build /app/public ./public

# The server opens "nexus.wal" relative to its working directory. Point it at the
# persistent data directory so the WAL survives container restarts.
RUN mkdir -p /app/data && ln -s /app/data/nexus.wal /app/nexus.wal

VOLUME ["/app/data"]
EXPOSE 8080

ENTRYPOINT ["./nexus_server"]
