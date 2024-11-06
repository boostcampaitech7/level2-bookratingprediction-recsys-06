source .env
scp -r $data_path server$server_name:/data/ephemeral/home/data/
ssh server$server_name << ENDSSH
cd /data/ephemeral/home/data
ls -lta
ENDSSH
./3.start-app.sh