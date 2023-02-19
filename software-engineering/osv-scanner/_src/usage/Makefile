scan.json: repos/app-playground repos/devhub;
	osv-scanner --json -r repos > $@

repos/app-playground:
	git clone https://github.com/vercel/app-playground $@ || (cd $@ && git pull)

repos/devhub:
	git clone https://github.com/devhubapp/devhub $@ || (cd $@ && git pull)

repos/nest:
	git clone https://github.com/nestjs/nest $@ || (cd $@ && git pull)

repos/next.js:
	git clone https://github.com/vercel/next.js $@ || (cd $@ && git pull)

repos/redash:
	git clone https://github.com/getredash/redash $@ || (cd $@ && git pull)
