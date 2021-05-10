console.log("hello")

chrome.runtime.onInstalled.addListener(function () {
    chrome.storage.sync.set({ color: '#3aa757' }, function () {
        console.log("The color is green.");
    });
    // declarativeContent: 見ているページのcontentsに応じたアクションをするためのAPI(ただし内容を読むことはない)
    chrome.declarativeContent.onPageChanged.removeRules(undefined, function () {
        // conditionsを満たした場合にactionsを起動する。
        chrome.declarativeContent.onPageChanged.addRules([{
            conditions: [new chrome.declarativeContent.PageStateMatcher({
                pageUrl: { hostEquals: 'developer.chrome.com' },
            })
            ],
            actions: [new chrome.declarativeContent.ShowPageAction()]
        }]);
    });
});