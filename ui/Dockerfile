FROM node:20

WORKDIR /ui

COPY package*.json ./
RUN npm ci

COPY . .

RUN npm run build:internal

CMD ["npm", "run", "start", "--", "-H", "0.0.0.0"]