#!/bin/bash
source .env

# 안 될 경우
# ssh server~ 접속
# cd /data/ephemeral/home
# git clone git@github.com:boostcampaitech7/level2-bookratingprediction-recsys-06.git
# 위의 순서로 실행
ssh $server_name << 'ENDSSH'
  cd /data/ephemeral/home
  eval "$(ssh-agent -s)"
  ssh-add /root/.ssh/id_ed25519
  echo "반드시 Github 개인 계정에 등록후 사용하세요 && ssh server로 직접 접속하여 실행"
  git clone git@github.com:boostcampaitech7/$prj_name.git
ENDSSH
echo "실행 완료! Press Any Key..."
read
