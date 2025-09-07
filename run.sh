#: <<'END'
echo
echo raefsky1
./gmres.o               matrices/raefsky1.mat 30 0.0000000001 300
./mp_gmres.o            matrices/raefsky1.mat 30 0.0000000001 300
./agmres.o              matrices/raefsky1.mat 30 0.0000000001 300
./mp_agmres.o           matrices/raefsky1.mat 30 0.0000000001 300
echo
echo raefsky2
./gmres.o               matrices/raefsky2.mat 30 0.0000000001 300
./mp_gmres.o            matrices/raefsky2.mat 30 0.0000000001 300
./agmres.o              matrices/raefsky2.mat 30 0.0000000001 300
./mp_agmres.o           matrices/raefsky2.mat 30 0.0000000001 300
echo
echo FEM_3D_thermal1
./gmres.o               matrices/FEM_3D_thermal1.mat 30 0.0000000001 300
./mp_gmres.o            matrices/FEM_3D_thermal1.mat 30 0.0000000001 300
./agmres.o              matrices/FEM_3D_thermal1.mat 30 0.0000000001 300
./mp_agmres.o           matrices/FEM_3D_thermal1.mat 30 0.0000000001 300
echo
echo cage11
./gmres.o               matrices/cage11.mat 30 0.0000000001 300
./mp_gmres.o            matrices/cage11.mat 30 0.0000000001 300
./agmres.o              matrices/cage11.mat 30 0.0000000001 300
./mp_agmres.o           matrices/cage11.mat 30 0.0000000001 300
echo
echo FEM_3D_thermal2
./gmres.o               matrices/FEM_3D_thermal2.mat 30 0.0000000001 300
./mp_gmres.o            matrices/FEM_3D_thermal2.mat 30 0.0000000001 300
./agmres.o              matrices/FEM_3D_thermal2.mat 30 0.0000000001 300
./mp_agmres.o           matrices/FEM_3D_thermal2.mat 30 0.0000000001 300
echo
echo stomach
./gmres.o               matrices/stomach.mat 30 0.0000000001 300
./mp_gmres.o            matrices/stomach.mat 30 0.0000000001 300
./agmres.o              matrices/stomach.mat 30 0.0000000001 300
./mp_agmres.o           matrices/stomach.mat 30 0.0000000001 300
echo
echo torso3
./gmres.o               matrices/torso3.mat 30 0.0000000001 300
./mp_gmres.o            matrices/torso3.mat 30 0.0000000001 300
./agmres.o              matrices/torso3.mat 30 0.0000000001 300
./mp_agmres.o           matrices/torso3.mat 30 0.0000000001 300
echo
echo cage13
./gmres.o               matrices/cage13.mat 30 0.0000000001 300
./mp_gmres.o            matrices/cage13.mat 30 0.0000000001 300
./agmres.o              matrices/cage13.mat 30 0.0000000001 300
./mp_agmres.o           matrices/cage13.mat 30 0.0000000001 300
echo
echo cage14
./gmres.o               matrices/cage14.mat 30 0.0000000001 300
./mp_gmres.o            matrices/cage14.mat 30 0.0000000001 300
./agmres.o              matrices/cage14.mat 30 0.0000000001 300
./mp_agmres.o           matrices/cage14.mat 30 0.0000000001 300
echo
echo cage15
./gmres.o               matrices/cage15.mat 30 0.0000000001 300
./mp_gmres.o            matrices/cage15.mat 30 0.0000000001 300
./agmres.o              matrices/cage15.mat 30 0.0000000001 300
./mp_agmres.o           matrices/cage15.mat 30 0.0000000001 300
echo