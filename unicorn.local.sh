export ROS_DOMAIN_ID=123  # car2 domain (car1=1)
export OPPONENT_IP=192.168.70.48  # car1 (opponent bridge target)

# 새 맵을 src에 넣으면 바로 읽도록(install 복사/symlink 불필요). runtime에서 src maps 직접 로드.
export STACK_MASTER_MAPS_ROOT=/home/nuc14/unicorn_ws/src/unicorn-racing-stack/stack_master/maps
