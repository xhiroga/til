# Ajaxとは
XMLHttpRequest(JavaScriptのオブジェクト)が全てのサーバーとの通信をする。

通常のリクエスト: フォームの値を全て送信する。
Ajax: DOMインターフェース経由(document.getElementByIdなど)で必要なフォームだけを取得して送信する。

通常のリクエスト: レスポンスが戻ってきたタイミングで、ブラウザが画面を再描画する。
Ajax: XMLHttpRequestのステートが4になったタイミングで,DOMインターフェース経由で画面の値を操作する。


# 参考
https://www.ibm.com/developerworks/jp/web/library/wa-ajaxintro1.html
