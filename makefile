run:
	nvcc -o saxpy saxpy.cu -run

profile:
	nsys profile --stats=true ./saxpy

clean:
	bash -c "rm ./report*"
