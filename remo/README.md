## Remo
Tool for image and annotation management.
* https://remo.ai
* https://github.com/rediscovery-io/remo-python for python API
* Django frontend, postgres db (`postgres://remo:remo@postgres:5432/remo`)
* python api is open source but actual remo app is not

## Usage
- Stop postgres if you are already running it on port 5432
- `docker-compose up -d`
- browse to [http://localhost:8123/](http://localhost:8123/)
- Files are placed in `./remo_home` and mounted in as a volume
- When you are finished: `docker-compose down`