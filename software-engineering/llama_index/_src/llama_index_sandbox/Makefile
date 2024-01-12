tmp/article-urls.csv: tmp/index.html;
	cat tmp/index.html | perl -nle 'print $& if m{(?<=href=")https://dev\.classmethod\.jp/articles/[^"]+}' > $@

tmp/index.html
	wget https://dev.classmethod.jp/tags/chatgpt/ -O $@
