To run things, first compile with ABAGAIL on classpath:

`javac -cp ABAGAIL.jar:. *.java`

Then for part 1 run

`java -cp ABAGAIL.jar:. NeuralNets | xargs python plot.py`
(this also requires matplotlib to be installed)

for part 2, run 

`java -cp ABAGAIL.jar:. FourPeaksTest`
`java -cp ABAGAIL.jar:. TravelingSalesmenTest`
`java -cp ABAGAIL.jar:. KnapsackTest`

this will provide results.

Charts will be in files named __ALGORITHM__plot.png
results.txt will contain results from part 1
