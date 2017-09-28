#!/bin/bash

echo "Confirm deletion of deeplearning products in this directory? (y/N)"
read decision
if  [ $decision == 'y' ] || [ $decision == 'Y' ] 
then
	DELETE_TARGETS="network_* fig* checkpoint* pretrain*"

	echo "Do you want to backup the products? (y/N)"
	read decision

	if  [ $decision == 'y' ] || [ $decision == 'Y' ] 
	then 
		BACKUP_FILENAME=`date +"%Y%m%d%R%S.tar.gz"`
		echo "Saving to $BACKUP_FILENAME"
		tar acf "${BACKUP_FILENAME} ${DELETE_TARGETS}";
	fi

	rm -rf ${DELETE_TARGETS};
fi

echo "DONE"

return 0

