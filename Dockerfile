# syntax=docker/dockerfile:1.6
#
# chora-adk-common Dockerfile — standalone Go library (no service binary).
#
# Build context = this repository. The library is consumed by the ADK crew
# services through the Go module proxy (github.com/apollo-chora/chora-adk-common),
# so there is no long-running process and no EXPOSE. This image exists to prove
# the module builds and its tests pass in a hermetic container, and to carry a
# warm module cache that downstream crew images can reuse as a build base.
#
# Shared Chora modules (chora-common, chora-contracts) are resolved through Go
# modules, not a workspace.

ARG GO_VERSION=1.26.6
ARG ALPINE_VERSION=3.23
ARG LIBRARY_NAME=chora-adk-common
ARG GIT_SHA=unknown
ARG BUILD_TIME=unknown

############################
# Stage 1 — build + test
############################
FROM golang:${GO_VERSION}-alpine${ALPINE_VERSION} AS builder

ARG LIBRARY_NAME
ARG GIT_SHA
ARG BUILD_TIME

WORKDIR /src

RUN apk add --no-cache ca-certificates git

COPY . .

RUN go mod download

ENV CGO_ENABLED=0 \
    GOOS=linux \
    GOARCH=amd64

# Build every package and run the suite. A library image that ships with a
# failing test is worse than no image: the failure would surface only in a
# downstream crew build.
RUN go build ./... && go vet ./... && go test ./...

############################
# Stage 2 — library base
############################
FROM golang:${GO_VERSION}-alpine${ALPINE_VERSION} AS library

ARG LIBRARY_NAME
ARG GIT_SHA
ARG BUILD_TIME

LABEL org.opencontainers.image.title="${LIBRARY_NAME}" \
      org.opencontainers.image.source="https://github.com/apollo-chora/chora-adk-common" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.created="${BUILD_TIME}" \
      org.opencontainers.image.vendor="Chora Platform" \
      org.opencontainers.image.licenses="UNLICENSED" \
      io.chora.library="${LIBRARY_NAME}" \
      io.chora.git-sha="${GIT_SHA}" \
      io.chora.build-time="${BUILD_TIME}"

ENV CGO_ENABLED=0 \
    GOFLAGS=-mod=mod

WORKDIR /src

# Source + the resolved module cache, so a downstream crew image can
# `COPY --from=...` this image or reuse the warm cache without re-downloading
# every dependency.
COPY --from=builder /src /src
COPY --from=builder /go/pkg/mod /go/pkg/mod

# Default action: re-verify the library. Override the command to build a
# downstream consumer on top of this image.
CMD ["go", "test", "./..."]
