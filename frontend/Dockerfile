# Build stage
FROM node:18-alpine as build

WORKDIR /app

# Install dependencies first (better caching)
COPY package*.json ./
RUN npm ci

# Copy source code
COPY . .

# Set production environment and build
ENV NODE_ENV=production
ENV PUBLIC_URL=/
RUN npm run build

# Production stage
FROM nginx:alpine

# Create necessary directories and set permissions
RUN mkdir -p /var/cache/nginx /var/run \
    && chown -R nginx:nginx /var/cache/nginx \
    && chown -R nginx:nginx /var/run \
    && chown -R nginx:nginx /var/log/nginx \
    && touch /var/run/nginx.pid \
    && chown -R nginx:nginx /var/run/nginx.pid

# Copy nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built assets from builder stage
COPY --from=build /app/build /usr/share/nginx/html

# Set permissions for nginx html directory
RUN chown -R nginx:nginx /usr/share/nginx/html \
    && chmod -R 755 /usr/share/nginx/html \
    && chown -R nginx:nginx /etc/nginx/conf.d

# Update nginx main configuration
RUN sed -i 's/user  nginx;//g' /etc/nginx/nginx.conf \
    && sed -i 's/user  nginx nginx;//g' /etc/nginx/nginx.conf \
    && sed -i 's/user  root root;//g' /etc/nginx/nginx.conf

# Switch to non-root user
USER nginx

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"] 