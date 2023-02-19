FROM gcr.io/datadoghq/agent:latest
RUN agent integration install -r -t datadog-lighthouse==2.1.0
RUN apt update \
    && apt install -y wget gnupg sudo \
    && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list' \
    && apt update \
    && apt install -y google-chrome-stable fonts-ipafont-gothic fonts-wqy-zenhei fonts-thai-tlwg fonts-kacst fonts-freefont-ttf libxss1 --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* \
    && wget -q -O - https://deb.nodesource.com/setup_16.x | sudo bash - \
    && sudo apt install -y nodejs\
    && apt clean \
    && apt autoremove \
    && npm install -g lighthouse@5.6.0

RUN npm install -g puppeteer --unsafe-perm=true\
    && groupadd -r dd-agent && usermod -a -G audio,video dd-agent \
    && mkdir -p /home/dd-agent/Downloads \
    && chown -R dd-agent:dd-agent /home/dd-agent \
    && chown -R dd-agent:dd-agent /lib/node_modules
