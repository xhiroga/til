# Event Stream

```shell
% rye run getting_started

% rye run firehose  # ログイン不要
MessageFrameHeader(op=<FrameType.MESSAGE: 1>, t='#commit') Commit(blobs=[], blocks=b':\xa2eroots\x81\xd8*X%\x00\x01q\x12 -\t\x11\x0c:?6T\x06\xc1+\x8app/i7\x82\xd2\x8e\xf9<\x16\x9c\xd4l6\x8f/\xdb\xbcigversion\x01\x8f\x01\x01q\x12 \x08\xf2\x92\x1c\xf3i\\\xde\x18Lm=\xcb\xd5#mI\xc8\xcc\xd4\xc5i\x95\xd2p\xec\xca\x91\x111\xb2\x8b\xa3e$typeuapp.bsky.graph.followgsubjectx did:plc:ctogwg3i3lglm35matp5hl5ricreatedAtx\x182023-07-01T20:21:21.551Z\xba\x03\x01q\x12 \x10\xd2\x10\x99\xda\x8aDJ\x1em\xb9\x11\xec\x1eh1\xe2\xb1E[\xe3\x1dk\xdf\xee\x87\x89\xc2\n\x1e\xab\xd2\xa2ae\x83\xa4akX app.bsky.feed.like/3jwo3eglbkk2jap\x00at\xd8*X%\x00\x01q\x12 \'\xf01\xcc)\x0b\xd7J\x8aVgQ\xc9\xa7\xec\xc8\xab\xa1\xaf\x15{\xcf\xbe\xa9\xa2sN\x93\xaf\x96\x19\xf6av\xd8*X%\x00\x01q\x12 \x9bjB\\\x18\x8cx4Y\x83 \xc8\x1bFk\xd8\xc3\x88?\xeb\xb0[\x06\xff\x0ct]\xd1kk!\x17\xa4akTrepost/3jw6pohdb7t2jap\x0eat\xd8*X%\x00\x01q\x12 .\x14\x99\x93\xdf\x85\x07\xb5\xa8Ol\x05\x97\xb4\x92r\x0eM\x9e\xf6\xc1\x8e\x04$\xda\x1aT\x9e\x12<\xdb\xa9av\xd8*X%\x00\x01q\x12 \xe2`%\xcf\x8f\x9a\x0f\'\xf9\x03\xe3o}\xa9b\x06W\xd5V\x98\xaa\x91\xdd\xd5\xdf\x93b\x89\x8e\xc4j\xe7\xa4akX\x1agraph.follow/3juixqlcmt22hap\tat\xd8*X%\x00\x01q\x12 \xcfQ\x1c\xb6\x07\x9b3\xeb\xbf\x97\x98\xc9\x933E\\\x8c\x06\xa8Z\\m7V\x07\x94s\xd3t].rav\xd8*X%\x00\x01q\x12 \x8c\x92\xbeY\xfe\xc9\x8f\xa4_\x85\xf4\x0b\x983\xd5\x8d\x1c\xda\xc4)\xf4p\x8d:\rP\xf4`R\x15\xa8\x00al\xd8*X%\x00\x01q\x12 \xe5&\xda4\xc6\xcc=\xca~\x97\\\xd6\x95\xc9\'c){\xe9/\xe1]\x8a\x85k\xd1\xba#\x1c\xe2\x99R\xd8\x06\x01q\x12 \xcfQ\x1c\xb6\x07\x9b3\xeb\xbf\x97\x98\xc9\x933E\\\x8c\x06\xa8Z\\m7V\x07\x94s\xd3t].r\xa2ae\x88\xa4akX#app.bsky.graph.follow/3jusgwnoei52map\x00at\xd8*X%\x00\x01q\x12 aa0\xb2\xf8Q\xec\xe4\x87\xd1\xf9\xe8\x117\x7f\xc2e4\xf7\x1b.\xc3\xb4\xbd\xf3H\x0c\xc7\xbbT3\rav\xd8*X%\x00\x01q\x12 x$\xc8T(Mq\xd7\xdcX6;\xc3h\xfc\x01#{\x04\xda\x1a\xac\x07r\xaf\xebC\x17e/\x1e\xda\xa4akKwo3fiqa622cap\x18\x18at\xd8*X%\x00\x01q\x12 \x9e\xf36\x17i\x13\xcfNU\xc6\x0e\x11o~\xbf\xd5[5\x83TB#\x94~\x97\xeb@nr2\x14\xefav\xd8*X%\x00\x01q\x12 TZe\xc1\xcc\x99|\x95\x80}{\xef\\C\x19\xa7\xbe\xf7Vh\x8ed\xaf\xae\xc9\xbe\x16"\x89~\xf0\x85\xa4akKz3wvpb5a32gap\x18\x18at\xd8*X%\x00\x01q\x12 \xd5\xd3\x04\xf6s{"v\xad\xc5\x12\xff0\'\xa0\xcb\xae\xde\xfa\xd0LZ\x1a4_\xc7\x92:\xda_\x86|av\xd8*X%\x00\x01q\x12 \xaa?\x13\xb2\x1a\xd3\xfb\'#x\xdf\xf0\x1b\x00[\xf7\x82\\M\x13<\xd2\x06\xdc\xf9\x044\xa9\xbf\x89}\xa6\xa4akJ4azowra42zap\x18\x19at\xd8*X%\x00\x01q\x12 \xac\xb3\xcf\xf86Jz1\xd8\xc6\xcb\x1c\xb0S@\xec\xa7V\xc2\xb5\xf3Bi\xcc%|\xfe\xd7\x9e\x93\xa9\x11av\xd8*X%\x00\x01q\x12 \xb8\x8f\x9c\xc1\xb7\xd6\x8c\xd2<e\xf4Vb\x8d\x96\x1cI)\xcf#;S\x91\xf9\xe2\xb7\xe6]\xef\xc8\xe6\x97\xa4akIb3ygbbm2xap\x18\x1aat\xd8*X%\x00\x01q\x12 \x87\xdfI\x94\xdd\x90\xad\xb6\xb5\x0f\xd6A\xbb\x90\xfer{\xa38\xad\xdc\xfe\xa0\xb3\xc7.\xae\n\x8b\x05\x1d\xdeav\xd8*X%\x00\x01q\x12 \xbf\xeb\x0e\xfe(d\xa3\xad\xedf\x89\xd2\x11\xcf"WC\x91M\xca\xfa\x0b\xe5V\xf7\xfaY\x15\x13\xd7\xbdJ\xa4akH4ajysc2pap\x18\x1bat\xd8*X%\x00\x01q\x12 \x01\xd1\xd8\x81\xba\x15[\xbbt\x83eW\x03\xdd^\x94L\x84\x06\xca$c\xd8\xd8\x87\xc2\x05x\x81a~eav\xd8*X%\x00\x01q\x12 R \xd1\xd7\xc1\xceZ\xa5G\x87\xea\xdeH4x\xfc;\n\x84\xe7J\xd8\x07\xd2\xba\x8d\x854C\xb7\x8e\xae\xa4akGuttpt2lap\x18\x1cat\xd8*X%\x00\x01q\x12 1\xd6\xa7\nA9\x9e!1W\x0f\x98*\x8c+\xa2\xe1\xf2\n:\x92\x10\xedS\x92?\x1d\xccD\x81\xd5\xf3av\xd8*X%\x00\x01q\x12 V\xe1\xc59\xf7\xae\xcc\x98;\xa2\xb1\x87\xcbr\x06\r\x06\x81\xaf\xf1\xca\x05\xd4\xfc\xb0\x80{\x86\xc4Z\x81\xe3\xa4akJieqsuppz23ap\x18\x19at\xf6av\xd8*X%\x00\x01q\x12 \x08\xf2\x92\x1c\xf3i\\\xde\x18Lm=\xcb\xd5#mI\xc8\xcc\xd4\xc5i\x95\xd2p\xec\xca\x91\x111\xb2\x8bal\xf6\xf6\x01\x01q\x12 -\t\x11\x0c:?6T\x06\xc1+\x8app/i7\x82\xd2\x8e\xf9<\x16\x9c\xd4l6\x8f/\xdb\xbci\xa5cdidx did:plc:rbrb64v2gmh7ctsoffqbpxxrcsigX@\x87\xd7S\xcaq\xd4$X~\xb2\xf2\xdd\x14B\x90I\x0bcJ\xac7\xb4\xf4\xc0|S\xaa\\\x03\x05q\xc9I\xb9\x1c\x07\x0f\xc6\x8d$W\x9c\x01\xcej\xf9m\x83\xb1\xa4v\x1b]\xe9P\x85\xfe~fH\xbe\xac\xa5Hddata\xd8*X%\x00\x01q\x12 \x10\xd2\x10\x99\xda\x8aDJ\x1em\xb9\x11\xec\x1eh1\xe2\xb1E[\xe3\x1dk\xdf\xee\x87\x89\xc2\n\x1e\xab\xd2dprev\xd8*X%\x00\x01q\x12 \x08\xee\xdf\x8c\xfai\xc5]7\xa6\xeb\x91\xb6x{\xed\x16\x92\xb6-\xb3\x85\xc4\xa5\x87;qW\x99\x17\xedggversion\x02', commit=CID('base58btc', 1, 'dag-cbor', '12202d09110c3a3f365406c12b8a70702f693782d28ef93c169cd46c368f2fdbbc69'), ops=[RepoOp(action='create', path='app.bsky.graph.follow/3jzieqsuppz23', cid=CID('base58btc', 1, 'dag-cbor', '122008f2921cf3695cde184c6d3dcbd5236d49c8ccd4c56995d270ecca911131b28b'), _type='com.atproto.sync.subscribeRepos#repoOp')], rebase=False, repo='did:plc:rbrb64v2gmh7ctsoffqbpxxr', seq=55706757, time='2023-07-01T20:21:35.663Z', tooBig=False, prev=CID('base58btc', 1, 'dag-cbor', '122008eedf8cfa69c55d37a6eb91b6787bed1692b62db385c4a5873b71579917ed67'), _type='com.atproto.sync.subscribeRepos#commit')
MessageFrameHeader(op=<FrameType.MESSAGE: 1>, t='#commit') Commit(blobs=[], blocks=b':\xa2eroots\x81\xd8*X%\x00\x01q\x12 a\x98\x98I\xe0\xe81\x89\xc8?\xd0\x16\xe5\xb5S/*,u<\x86G\xf3\xbb\xd5xM\xd6\t\xf0\r\xcdgversion\x01\xf8\x01\x01q\x12 \x98\xc3Y2\r\xd5u\xd3\xcb\'\xd8$\\\'\x00#\x1f\xaa\xe0Y\x88\xe7o\x95P\\\xb1\xa6\xa2\'\xcc\x04\xa3e$typerapp.bsky.feed.likegsubject\xa2ccidx;bafyreihlhuvdifycsvlfji5wdiaa66v73gshqoge6zburcqvurlq2hmrhmcurixFat://did:plc:kqcmjyv4zxfcxzbitkw7z2d6/app.bsky.feed.post/3jzicu2xx352picreatedAtx\x182023-07-01T20:21:15.301Z\xd3\x01\x01q\x12 F\xfe\xcc\xb9p~\r\x05\x8c\x90\xd6]\xb0\x8f\xb2U5\x06_\xb2=\xe9\xf6\xfaX\xe1+\x1d\xfd\x15\x89\'\xa2ae\x81\xa4akX"app.bsky.graph.block/3jzfbozyzrl2eap\x00at\xd8*X%\x00\x01q\x12 \xecK\x11\x1b\x84{W\xf2Z\xe95T\xe8\xae\xef6\xe3\x84R#G[\xe6\xf7\x7f\xd9\xfa78\x9c\xba\xf1av\xd8*X%\x00\x01q\x12 \xd8\xad\xe7]\x83a\xa9MW\xbd.Wn\x9c\xb9\x01\x01\xa4\xaf\xfa\xc8\x7f\xee\x14\xf6C\r[\x1b4\xff\xe5al\xd8*X%\x00\x01q\x12 \x81\r4\x07t\x82\x9d\xde]M\xb0Z\x1b#\x15\x83\xd6\xef\xd8a\x07`W9=aj+\x89A\xdeH\x9f\x03\x01q\x12 \x81\r4\x07t\x82\x9d\xde]M\xb0Z\x1b#\x15\x83\xd6\xef\xd8a\x07`W9=aj+\x89A\xdeH\xa2ae\x83\xa4akX app.bsky.feed.like/3jxj3gfnbsa2bap\x00at\xd8*X%\x00\x01q\x12 \xd0\xb9\xa5\xe8\xd4\xba\x9e\xc1&\xb1)\xfaq\xea\xe8\xcd\x02\x07\x07\x19\x8f\xf0\xca\xca@\xbc"\xca\x9aS\xc6\xadav\xd8*X%\x00\x01q\x12 \xcf\xe2\\@\x95\x08\xb9!V\xca\xfe\x96\x08\xd1\xe7\'\x9a1\xb5\x8a\xbdn\xa2\x83g\xf1\x85\x15y\xb4\x8e\xc7\xa4akHoe2xye2hap\x18\x18at\xd8*X%\x00\x01q\x12 \xb0\xdb)>\x90\xe8\xecX\x99\x18>\x08\x1e\xaa(\x04\xa1\xa0\xb4\xc0\x97\x1f\xa8\xa1\xbc\xbfQ{\x08:\xff\xaaav\xd8*X%\x00\x01q\x12 Me)\xb3\x8e\xf5AI\x8e\xdf\x93\x95\x13\x0c\xfe\xe0\xf9\x10\xc9\x15\xfc\xd1\xf4\xf3\xf3\xda\x01w\x8f[\xdf6\xa4akKyhjrxx2lr2tap\x15at\xd8*X%\x00\x01q\x12 1\x13\xf4]\x95\x1e!@\xc5\xdc\xbf-\xf9s\x96\xe9\x92\xf5\x90\'\xd1\xfb\xb1\xb0`\xbb\xb7\xab\x04h\xc1\x00av\xd8*X%\x00\x01q\x12 {\xc3\x0e\x82\x7f\xdbu\x0b\x87\xb8L\x001\x9cn\xa4W\xeb6\xf3\xb1\x1c%\x8a\xea\xd1\xa5bGh{\xbfal\xd8*X%\x00\x01q\x12 \x7f\xa0\xa3#\x99\xb2J\xecg\xfe^\t\xda\x15\xf2\xfb\x1e\x8a\xd0\xd3\xbd\xcb\xf43\xf1a\xf5`I\xe4\xf6\xf4\x95\t\x01q\x12 1\x13\xf4]\x95\x1e!@\xc5\xdc\xbf-\xf9s\x96\xe9\x92\xf5\x90\'\xd1\xfb\xb1\xb0`\xbb\xb7\xab\x04h\xc1\x00\xa2ae\x8a\xa4akX app.bsky.feed.like/3jymx4dltib25ap\x00at\xd8*X%\x00\x01q\x12 \xf86\x07\xb2\xe1\x9a9\x82e\xfaN\xe4i\xf3\x99\x84P\xfabtlI\xb6"K\xa6\n\xcea\xd3\xb1\xf1av\xd8*X%\x00\x01q\x12 \xda\xbe\x0c\x14\x04U1\x06\x87\xc6\x81uCF\xcb\x1b\xcd\xb0\xe2\x10@2?\xc7$\xab\x0fg\x15\xae\xca=\xa4akJni63xfc22lap\x16at\xd8*X%\x00\x01q\x12 vQE\x8b|b\x15\x98\x17\xcc\xe7\x8f\x8c&x\x1b\x8c\xa9)\xdc\x9a\xe2\x14\xc4\xc0\xe8\xd1\xf0\x8d\x1a\x1e\x86av\xd8*X%\x00\x01q\x12 \x81\x16IK@\xd9[\xf4\xeeg@\x99+\xa6\xa1\xf2\xd7%\xb0\xd2i\x925a\' \xaeV\xc0\xbfH\'\xa4akKz6tb4dkdr2tap\x15at\xd8*X%\x00\x01q\x12 \xa5\x7f\\\xe9\xa4\x9a\xbd\xa4\xe8\xb5\xb1\xa9\x1d\xf0\x86\xe3\x9ak\xfa\\-\xc2\x05\x0f<\xad\xbdA\x90N\xe7\x93av\xd8*X%\x00\x01q\x12 \x91:\xe5\xb8\xce\x18)\x05\xf0\x0e\x8b\xde\x80\xb2\xe4{S\xda\t\xeaR\xa4\xd5\x04b\xb4\x08;\xda\x11{j\xa4akRpost/3jwifrbyxbs25ap\x0eat\xd8*X%\x00\x01q\x12 \xce\xe8@\xbb?\xc3\x9c\x81\xdc\xa41~h\x08\xae\xf12;]\xb2\n\xd6\x009iA\x93\xcf\xcd\x89?Dav\xd8*X%\x00\x01q\x12 \xc5\xbb\xb6\xfdu\n\xb1\x1e\x95\xb2\xe7d\xbf\x1cp\xe8r\xe5\x96\xd6\x1d\xf0%,\xcd\x9c\xcd\x05AG\xb3B\xa4akJo4fu7fec22ap\x16at\xd8*X%\x00\x01q\x12 \xb7\x9a4\x9b%[\xdc\xb3{.\x9797\xef%d\xe6L\xef\x98\x90v\xc5Nds\xcd\xbfJ\xc3U\xbbav\xd8*X%\x00\x01q\x12 T4\x01}\xc1\xcf3\x86\xac\xf5\xc4\xb44\xc4\x8e\xf0\xb3\x11\xa7\x83\xe7)P\xf23\xbcp\x08\xbe\xf7\xca\xad\xa4akTrepost/3jwkvtlf6w22dap\x0eat\xd8*X%\x00\x01q\x12 \xb93!Z\xc39\xc7\xae2I\xd0\xa8\x8f\xa9:\xba\xa81#\x14\x91\xb5\xafR\xa0r\xb5\xd0\tm\x14\xe8av\xd8*X%\x00\x01q\x12 \xb2]\xe5~\xd3V\xa4%\x97\x1cp\xcc\x07`o\x0c\x9f\xc0\xea=\xde\xb2\xa3\xe4\x0e9\xce\xb7\x01Q\x17N\xa4akX\x19graph.block/3jwdjr5hy6427ap\tat\xd8*X%\x00\x01q\x12 \x93`X\xb8\xf2\xaf\xf7\x13\x9b\xfb\'Ox\xe4rI\x9f\x03\xf1\x9d\xdc~\x02\x0fMxR\xce9\xbb\xa3`av\xd8*X%\x00\x01q\x12 I\xebCc\xaaa\xe0b\x12^yQ\x87z\xc0I|P\xea1y\x19\x91@P_\x0e5\x99W\xf1\xcd\xa4akIqjuby7m2xap\x18\x19at\xd8*X%\x00\x01q\x12 \x8f-\xe4\xab\x1el\x91:5=m\xe4vH\x14|*\x9b\x97\x13F\x1a,\xd0\x9f\x12[\x1e=w\xe6\xd2av\xd8*X%\x00\x01q\x12 \xfe\xb0\xf7b\xd4Y\xe0\x94\xc2\xc0\x97\xad\x87\xd8\xa8u\xa4\xa5\xff\xa4\xfbc\xec\x08\x1a\x1d0\x943\xdc\xfdk\xa4akJkvs63fec2dap\x18\x18at\xd8*X%\x00\x01q\x12 #,\xfa\x12w\xb7P[\t`\xa9\xa7(#m\xb8\xa7m*\x03_\xe7\xb8\x1a\x7f\x16y\x100\xde\xd3\xa3av\xd8*X%\x00\x01q\x12 \xb2\x9a\xb1ab\x81o/\'Y\x9f\x005\xa8\xf1\x1d\xcaP`\xcf&\xf9\xd9\xacK-\x18@\xd1\xb8\x197\xa4akKym6a65hbj2yap\x17at\xd8*X%\x00\x01q\x12 \xdf\x19\x1a\xff\xd2t\x85\x06\xd2L-e\xba%e}I\xf4\xa1\xc2\xa1\x92}-\xdfJ$p\xf7Q(hav\xd8*X%\x00\x01q\x12 /~\xa6E\x14\x94H\xb0\xb1k,\xf9\xbe\xae_,\x19\xf3\xbf\x9eHZH\xb9\xa6\xeblb\xb1/\xcf)al\xd8*X%\x00\x01q\x12 K\x87\xc2\xdd\xbcji9\xcd?sKQ\x92?\x0e\x05\xfdX\r\xd8^\x92\xbb\xd5)\x85\x8ehV,7\xb6\x03\x01q\x12 \xa5\x7f\\\xe9\xa4\x9a\xbd\xa4\xe8\xb5\xb1\xa9\x1d\xf0\x86\xe3\x9ak\xfa\\-\xc2\x05\x0f<\xad\xbdA\x90N\xe7\x93\xa2ae\x84\xa4akX app.bsky.feed.post/3jvbr2wzn2x2dap\x00at\xf6av\xd8*X%\x00\x01q\x12 \xc8e\x12Z\xb6\xaaX\xe4\x86x\xc3\xf4\xd2\xd1#R\xca\xfeL}1\x10\x1a\xa5\xb8El\x1f\xac\xdeA\xcb\xa4akJdnkuyb2n2hap\x16at\xd8*X%\x00\x01q\x12 )\xa4cK\x01\xd9\xfe\xbeT\xe2\xd3U\x00\x7f\xe2|\x12nX\xa5+\xc0X:S\x10\x12\xcfx\x05A1av\xd8*X%\x00\x01q\x12 F\xa8\xf3\xdaN\xc4\xd2\x82\x85$\xfa*\x96,\x8ez\xb7j\x15u\xef\xf34v0!/\x01\x0c\x1a\x8f\xcb\xa4akKwdjplj53d2rap\x15at\xf6av\xd8*X%\x00\x01q\x12 \x1521s\x83\xe4\xf6\xb78\xd3\x87\'"LM\xbaY61\x81\xb0\xb4\xfe\x85\xc4\'\xcb.\xd5\xe5Z\xc2\xa4akHzoxiyl2rap\x18\x18at\xd8*X%\x00\x01q\x12 \xa9\xf9X\xcb+\x93\xdd8n\xc7\r\x97\xd6\xb2\x1f(\x8d\xdf~\x86\xaa(;\xd2>T\xc6\xe8+\xa5P\xa5av\xd8*X%\x00\x01q\x12 \xf8\xf4\xd42$\xec\x1c9\xd0\xcfln\xd0\xe8\xb8\xbfv\xab\xa61\xd7L\xf5K\x98d\xf5\xc8\x9b0\x08\xa2al\xd8*X%\x00\x01q\x12 m|;a\xb5\x869p\xc5\x94\xb4{\xe3\xe7\x18\xc4h\xbd&q2P;\x87\xfa\x91\xe6B03^>\x84\x03\x01q\x12 m|;a\xb5\x869p\xc5\x94\xb4{\xe3\xe7\x18\xc4h\xbd&q2P;\x87\xfa\x91\xe6B03^>\xa2ae\x85\xa4akX app.bsky.feed.like/3jzieqiwh572oap\x00at\xf6av\xd8*X%\x00\x01q\x12 \x98\xc3Y2\r\xd5u\xd3\xcb\'\xd8$\\\'\x00#\x1f\xaa\xe0Y\x88\xe7o\x95P\\\xb1\xa6\xa2\'\xcc\x04\xa4akRpost/3jv4lgn6quf2zap\x0eat\xf6av\xd8*X%\x00\x01q\x12 \x0c^\x977\xac\r\xbe\xad\x05\xc0(L\xaa\xbc\xcd\x16\x1f3yT\xaay\xdd\xf7\xf3\x8f\x81\xc9u\xeb\x11{\xa4akJ7zqn6be32zap\x16at\xf6av\xd8*X%\x00\x01q\x12 \xf6\xa4|\x1c\xf6\xd5:\xafC\xb3?\x7f\xcd\xad\x17\xab\xfc\xe3\x02<+\xe4aq\xd5\xc4\x1eb\xc2$\xb5\xf9\xa4akHvsaddo2uap\x18\x18at\xf6av\xd8*X%\x00\x01q\x12 "\xce/\xbd\x96\xfe\x13*f\x0f\x17\xf90\x03[\xf4\xf9#\xf9\x83@\xe7\xa2\xf6W\xc8\xbd\xb4\xb5\xb7\xecE\xa4akJartgouc62tap\x16at\xf6av\xd8*X%\x00\x01q\x12 \xd7e\xb8&zEc\xc4\xbb\x0e\xecwx\x026\xa6\x8a\xc7\xd4\x1a\x8e\xf7\xfb\x1c\'\x12\x10\x94\xab\t\x802al\xf6\xf6\x01\x01q\x12 a\x98\x98I\xe0\xe81\x89\xc8?\xd0\x16\xe5\xb5S/*,u<\x86G\xf3\xbb\xd5xM\xd6\t\xf0\r\xcd\xa5cdidx did:plc:5t7usit4fbkpbzklvcuziqnlcsigX@\xfe\xdeFf\xf0\xa7\x8fOq[\xe1\xdf\xc8\xa6\xb97\xaeZ\xaa!\xc5N\x8f\x971v\xdcX\x00\x96\x95\x88(\x84\xe9\xc9>H\xf5\xa1>N7W\xee\xa5\xa6\xa99\r?\xc7x\xec9z\xb1\x9e\xc6\xfbL\xa9\x7f\xf6ddata\xd8*X%\x00\x01q\x12 F\xfe\xcc\xb9p~\r\x05\x8c\x90\xd6]\xb0\x8f\xb2U5\x06_\xb2=\xe9\xf6\xfaX\xe1+\x1d\xfd\x15\x89\'dprev\xd8*X%\x00\x01q\x12 \xa6Kq\x9c\x82\x1d\xc4\xd7QxX\xb4\x12\x82V\x9e\xd2f\x02\xb2\x8a\xa91(~d\xde\xe7\xdd\xac\xc7\x8dgversion\x02', commit=CID('base58btc', 1, 'dag-cbor', '122061989849e0e83189c83fd016e5b5532f2a2c753c8647f3bbd5784dd609f00dcd'), ops=[RepoOp(action='create', path='app.bsky.feed.like/3jzieqiwh572o', cid=CID('base58btc', 1, 'dag-cbor', '122098c359320dd575d3cb27d8245c2700231faae05988e76f95505cb1a6a227cc04'), _type='com.atproto.sync.subscribeRepos#repoOp')], rebase=False, repo='did:plc:5t7usit4fbkpbzklvcuziqnl', seq=55706758, time='2023-07-01T20:21:35.665Z', tooBig=False, prev=CID('base58btc', 1, 'dag-cbor', '1220a64b719c821dc4d7517858b41282569ed26602b28aa931287e64dee7ddacc78d'), _type='com.atproto.sync.subscribeRepos#commit')
MessageFrameHeader(op=<FrameType.MESSAGE: 1>, t='#commit') Commit(blobs=[], blocks=b':\xa2eroots\x81\xd8*X%\x00\x01q\x12 \xcb@\xd8.{\x83\xe5\x00\xb3\xcb.\xd2\x11\x1b\x93\x07k]+\xf6B\xa6\xde\x08gh\xf4\xd3\xc9\xc53\x1dgversion\x01\x8f\x01\x01q\x12 \xd0}rX\xf2\x96p\x9d\xde\x8a\xbb\xa7\x8e\xa9~\xe5\x1d\x82\xe2\xe41;%\xa8D\n\x14\xe1 8\x8b\xea\xa3e$typeuapp.bsky.graph.followgsubjectx did:plc:nwnlnixtjh3qhkwpz2uy5uwvicreatedAtx\x182023-07-01T20:21:30.147Z\xd4\x01\x01q\x12 {z=\xbbp\n\xbd"j9\x1c\x12i\xbcB-@\x91\xe7\x8d;R\x03G\xd2\xeb\xff\xa6\xd8\x81\x05\x08\xa2ae\x81\xa4akX#app.bsky.graph.follow/3jziepfdfxk2qap\x00at\xd8*X%\x00\x01q\x12 \x0f+\x03\x99\x01\x9eSH<\x8f\xb7\xc0\xccV\xdfes\x96\xe4\xc6\xb4\xa9k\x80\x000\x02\xb9\xed\x02N\\av\xd8*X%\x00\x01q\x12 \x06\xfb\xa3r#\x1c\xbeE\x92RS2F\xebT\x9f\xbb\r\xa0V4|\xcf\x85\xb3\xe1s\xca\xb7\x96\x1a\xa8al\xd8*X%\x00\x01q\x12 WB>Q\x95\xcc\xd5\xe3n~z\x1b7\x9a\xa0\xc1e\xc5\xa2\xde\xa8\xf3{2 \xa2y\xc9\x1b&;\x08S\x01q\x12 \x0f+\x03\x99\x01\x9eSH<\x8f\xb7\xc0\xccV\xdfes\x96\xe4\xc6\xb4\xa9k\x80\x000\x02\xb9\xed\x02N\\\xa2ae\x80al\xd8*X%\x00\x01q\x12 \x0c\xf0\x02\x19^\x9a\x7f\xe3[\xff\x9a4b\xec%\x06l6I\xfe\x81\n\x0b\xa8M\xfc/\xa6_\x83`W\xbc\x02\x01q\x12 \x0c\xf0\x02\x19^\x9a\x7f\xe3[\xff\x9a4b\xec%\x06l6I\xfe\x81\n\x0b\xa8M\xfc/\xa6_\x83`W\xa2ae\x84\xa4akX#app.bsky.graph.follow/3jziepnmxym2fap\x00at\xf6av\xd8*X%\x00\x01q\x12 `9\xed\xff\xc2#\xb6g\xc7\x98\xa4\xa7\xfb\xd7\xee9\x1f\xd1()\x9f\xf9\xbe\x81\xf5\x7f[\x0b\x12+g\x8d\xa4akHqcj4te2cap\x18\x1bat\xf6av\xd8*X%\x00\x01q\x12 \x8f\xde\xe9\xef\x8f7r.m\x12BN\xb5\x19\xbb\x81\x83\xcc\x07\x08\xb1\xb9\\%\x01\x92a\x93\x9dp\x14\xcf\xa4akGevysu2fap\x18\x1cat\xf6av\xd8*X%\x00\x01q\x12 mJ\xc7\x94U\xe9\xfd\xe27\xaa\xa4\xd6\x94\xde~\'k\x02\x00V\xfb\xad\x0b\x96\x85\x02\xe0\xc3\xa39\x0c7\xa4akGwyuaz2fap\x18\x1cat\xf6av\xd8*X%\x00\x01q\x12 \xd0}rX\xf2\x96p\x9d\xde\x8a\xbb\xa7\x8e\xa9~\xe5\x1d\x82\xe2\xe41;%\xa8D\n\x14\xe1 8\x8b\xeaal\xf6\xf6\x01\x01q\x12 \xcb@\xd8.{\x83\xe5\x00\xb3\xcb.\xd2\x11\x1b\x93\x07k]+\xf6B\xa6\xde\x08gh\xf4\xd3\xc9\xc53\x1d\xa5cdidx did:plc:3yg3xncri4eu6sx3cuniedrwcsigX@\x87\xb7\x10\xa8\x03\xdb[\xe3\xf5\xc3\xa2Q\xeb>e\xdcY\xb1*\xf7j\x8eI\xb6x\x1f\xba\xb85\x8f\x054I\xe6\x94C\xa8\x9d\xe3:K\xb2\x9c\x81&\xfa\xd47\xbar\x9c?\xd1)\xa9\xd1K!\xb2\xe2p<\x84\xfdddata\xd8*X%\x00\x01q\x12 {z=\xbbp\n\xbd"j9\x1c\x12i\xbcB-@\x91\xe7\x8d;R\x03G\xd2\xeb\xff\xa6\xd8\x81\x05\x08dprev\xd8*X%\x00\x01q\x12 N\x81=\xfbj}\xa3\x92\xc1\x8d~+\xbdv-\x93\xe1\r\xcb\xa97\x13\xbb\xfe\x08;y\xacY\x1c\xbd\x9bgversion\x02', commit=CID('base58btc', 1, 'dag-cbor', '1220cb40d82e7b83e500b3cb2ed2111b93076b5d2bf642a6de086768f4d3c9c5331d'), ops=[RepoOp(action='create', path='app.bsky.graph.follow/3jzieqwyuaz2f', cid=CID('base58btc', 1, 'dag-cbor', '1220d07d7258f296709dde8abba78ea97ee51d82e2e4313b25a8440a14e120388bea'), _type='com.atproto.sync.subscribeRepos#repoOp')], rebase=False, repo='did:plc:3yg3xncri4eu6sx3cuniedrw', seq=55706759, time='2023-07-01T20:21:35.699Z', tooBig=False, prev=CID('base58btc', 1, 'dag-cbor', '12204e813dfb6a7da392c18d7e2bbd762d93e10dcba93713bbfe083b79ac591cbd9b'), _type='com.atproto.sync.subscribeRepos#commit')
```

## References

- [Event Stream | AT Protocol](https://atproto.com/specs/event-stream)