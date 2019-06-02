# Deep Cost-sensitive Kernel Machine Model 
This is an implementation of the Deep Cost-sensitive Kernel Machine (DCKM) model described in the **Deep Cost-sensitive Kernel Machine
for Binary Software Vulnerability Detection** paper.

DCKM model is a combination of a number of diverse techniques, including deep learning, kernel methods, and the new cost-sensitive based approach, aiming to detect efficiently potential vulnerabilities in binary software.

The overall structure of DCKM model consists of 3 primary elements: 
- an embedding layer for vectorizing machine instructions.
- a Bidirectional Recurrent Neural Network capable of taking into account temporal information from a sequence of machine instructions.
- a novel Cost-sensitive Kernel Machine invoked in the random feature space to predict the vulnerability with minimal cost-sensitive loss.




