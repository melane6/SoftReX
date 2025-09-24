# SoftReX - Causal Responsibility EXplanations for Software Faults

### Work in progress

How to represent the code in the most effective way for fault localization and explanation?
- [ ] symbolic analysis -clear(c++) & angr **version**
- [ ] simple lines of code
- [ ] ast analysis

Types of graphs:
- [ ] control flow graph
- [ ] data flow graph
- [ ] program dependence graph

Mutation strategies:
- Simple Remove Lines/Blocks of code/Branches of trees
- Changing Operators -> arithmetic, logical, relational
- Mutating IF-ESlE statements and loops
- Check other papers for other strategies for software fault localization

#### Plan
- Implement three different types of representation of code
- Create a compatible graph/tree representation for each of them
- Use this graph/tree representation to compute causal responsibility explanations for software faults
- Compare the effectiveness of each representation in terms of fault localization and explanation quality
- Ablation Study to see the effectiveness of each representation and mutation strategy

#### References
- [Applications of Causality and Causal Inference in Software Engineering]
- [Systematic bug finding and fault localization enhanced with input data tracking]
- [Testing Causality in Scientific Modelling Software]
- [An effective fault localization approach based on PageRank and mutation analysis](https://www.sciencedirect.com/science/article/abs/pii/S0164121223001942)
- [Large Language Models in Fault Localisation](https://arxiv.org/abs/2308.15276)
- [A Systematic Exploration of Mutation-Based Fault Localization Formulae](https://doi.org/10.1002/stvr.1905)
  - https://github.com/TinyKingKong/MBFL-study