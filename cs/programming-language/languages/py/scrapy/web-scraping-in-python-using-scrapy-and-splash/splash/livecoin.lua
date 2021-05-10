-- url = https://livecoin.net/en
function main(splash, args)
    splash.private_mode_enabled = false     --default true
    url = args.url
    assert(splash:go(url))
    assert(splash:wait(1))
    tabs = assert(splash:select_all(".filterPanelItem___2z5Gb"))

    litecoin_tab = tabs[5]
    litecoin_tab:mouse_click()
    assert(splash:wait(1))
    splash:set_viewport_full()

    return splash.html()
end
