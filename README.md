# wash-mark-checker-lolipop
lolipopデプロイ用  
画像認識による洗濯表示の識別

## Description
画像から洗濯表示を識別し、意味を表示する

>洗濯表示とは？  
衣服についたタグのマークのこと。  
洗濯や乾燥の方法、アイロンのかけ方やクリーニングの方法などが示されている。  
全41種類。世界共通。

- 洗濯表示をまとめて調べる  
![screencapture-wash-mark-checker-lolipop-io-2019-12-20-20_33_43](https://user-images.githubusercontent.com/20394831/71253446-45ef3980-236b-11ea-896a-d3e3052f4e51.png)

- 洗濯表示を個別に調べる
![screencapture-wash-mark-checker-lolipop-io-individual-2019-12-20-20_34_01](https://user-images.githubusercontent.com/20394831/71253445-45ef3980-236b-11ea-9452-7a586cfb24eb.png)


## Features
- 洗濯表示の識別
  - まとめて識別
  - 個別に識別
- 意味出力

## Deploy
1. ロリポップ！マネージクラウドのアカウントを作る
2. pythonでプロジェクト作成（pythonのバージョンは3.6）
3. sshを設定
4. ロリポップとローカルを結ぶ ![](https://support.mc.lolipop.jp/hc/article_attachments/360018389253/python-ssh-info.png)  
画像のリポジトリを叩く
5. ローカルからロリポップへpush  
`git push lolipop master`
6. ssh接続  
4の画像のSSHコマンドを叩く
7. 重みをダウンロード
ロリポップにて
`wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1xYyucyZy3xew9jkdQk9NCa559Qvmnwg3' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1xYyucyZy3xew9jkdQk9NCa559Qvmnwg3" -O trained_weights_final.h5 && rm -rf /tmp/cookies.txt`

`mv trained_weights_final.h5 current`

8. pythonの環境を整える  
ロリポップにて  
`mkdir -p bigass/space`  
`export TMPDIR=bigass/space`  
`pip install --user -r current/requirements.txt`

9. ロリポップコンソールにて、起動コマンド変更
![](https://support.mc.lolipop.jp/hc/article_attachments/360017672594/command-input-flask.png)
`/var/app/shared/bin/gunicorn --bind=0.0.0.0:8080 -t 300 --chdir=/var/app/current app:app`  
コンテナ再起動

10. 表示確認
少し時間かかる
