# Build stage
FROM golang:1.24-alpine AS builder

WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the api-server example
RUN CGO_ENABLED=0 GOOS=linux go build -o /api-server ./examples/api-server

# Runtime stage
FROM alpine:latest

RUN apk --no-cache add ca-certificates

WORKDIR /app

# Copy the binary
COPY --from=builder /api-server .

# Expose port
EXPOSE 8080

# Run the server
CMD ["./api-server"]
