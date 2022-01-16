# Chrome Extension

Chromeを使って動くアプリケーション。内部的にはExtensionのためのページを開いているような扱いとなる。  
chromeクラスから各種APIが使用可能(import不要)  


# Usage
開発中のExtensionの追加方法:  
chrome://extensions/ で Developper modeをオンにし、LOAD UNPACKEDから選択する。  

# TIPS
* chormeのAPIがundefinedになっている場合、manifestでpermissionを設定していないかもしれない。  
* ドキュメントにはアイコンにJPEGが使えるとあるが、実際は使えない（試した限り）

# Reference
https://developer.chrome.com/extensions/getstarted
[Sample Extensions](https://developer.chrome.com/extensions/samples)