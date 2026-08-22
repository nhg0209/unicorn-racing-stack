# 2026-08-20 맥 차(car1)와 통신하려고 도메인을 1로 통일. 맥은 unicorn.sh 기본값 1을 쓴다.
# :- 로 두어 셸에서 미리 export 하면 그 값이 이긴다 (nuc14 붙일 땐 ROS_DOMAIN_ID=123).
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export OPPONENT_IP=192.168.70.48  # car1 (opponent bridge target)

# 새 맵을 src에 넣으면 바로 읽도록(install 복사/symlink 불필요). runtime에서 src maps 직접 로드.
export STACK_MASTER_MAPS_ROOT=/home/nuc14/unicorn_ws/src/unicorn-racing-stack/stack_master/maps
