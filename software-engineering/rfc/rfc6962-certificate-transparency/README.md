# Certificate Transparencyを知ろう


## 歴史的経緯

ドメインの持ち主以外が証明書を勝手に発行する事例が背景となり、Googleのエンジニアが2011年に提唱した。証明書の発行をログサーバーに届け出ることで、ドメインの持ち主は不正な届け出を検知できる。

知人のベテランエンジニアの方から聞いたストーリーは興味深かったが、私かその人が誤解しているようだった。  
「シマンテックが `google.com` の証明書を内部で勝手に発行したので、その対策としてGoogleが考えた仕組みである。」これは時系列的に誤り。  

## 問題

- CTサーバーが中央集権的に管理されているため、ハッシュ再計算により追記のみの前提を覆せるという指摘がある。
- CTサーバー管理者は全世界のアクセス情報を知ることができる。
- IPアドレスを全件クロールするのに比べ、圧倒的に速くドメイン情報を収集できる。`admin.example.com` のようなドメインが、登録直後に簡単に発見されてしまう。

## CTで遊ぶ

- 検索: https://crt.sh/
- リアルタイム監視: https://certstream.calidog.io/
- 統計: https://ct.cloudflare.com/

## Reference

- [Certificate Transparencyを知ろう ～証明書の透明性とは何か～](https://www.jnsa.org/seminar/pki-day/2016/data/1-2_oosumi.pdf)
- [Certificate TransparencyによるSSLサーバー証明書公開監査情報とその課題の議論](https://www.slideshare.net/kenjiurushima/certificate-transparencyssl)
- [電子証明書の透明性](https://ja.wikipedia.org/wiki/%E9%9B%BB%E5%AD%90%E8%A8%BC%E6%98%8E%E6%9B%B8%E3%81%AE%E9%80%8F%E6%98%8E%E6%80%A7)
