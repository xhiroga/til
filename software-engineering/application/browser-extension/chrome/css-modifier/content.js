const getElementByXpath = (path) => {
  return document.evaluate(
    path,
    document,
    null,
    XPathResult.FIRST_ORDERED_NODE_TYPE,
    null
  ).singleNodeValue;
};

function getAccountId() {
  return getElementByXpath('//*[@data-testid="aws-my-account-details"]')
    .innerText;
}
function getHeader() {
  return getElementByXpath('//*[@id="awsc-nav-header"]');
}

const getItem = async (key) =>
  new Promise((resolve) => {
    chrome.storage['local'].get(key, resolve);
  });

const loadConfig = async () => {
  const config = (await getItem('config')).config; // '[{"accountId": "123456789012","color": "#377d22"}]'
  return JSON.parse(config);
};

const selectColor = (config, accountId) => {
  return config.find((color) => color.accountId === accountId);
};

const patchColor = (color) => {
  const headerElement = getHeader();
  headerElement.style.backgroundColor = color.color;
};

const run = async () => {
  const config = await loadConfig();
  const accountId = getAccountId();
  const color = selectColor(config, accountId);
  patchColor(color);
};
run();
