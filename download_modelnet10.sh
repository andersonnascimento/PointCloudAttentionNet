#/bin/bash

![ -d "ModelNet10" ] || curl -o ModelNet10.zip http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
![ -d "ModelNet10" ] || unzip ModelNet10.zip
