# Jinja2

Pythonのテンプレートエンジン。  
そのまま使用する場合はEnvironmentインスタンス→templateインスタンス→render()でhtmlをレンダリング。  
Flaskと併用する場合はrender_template()メソッドにテンプレと変数をそのまま渡せばOK。  


* 引数はキーワード引数、dict型のどちらでもOK、ただしFlaskのrender_template()はキーワード引数だけ!
template.render(name="Hiro") # OK
template.render({"name":"Hiro"}) # OK
return render_template("index.html", user={"name":"Hiro"})

