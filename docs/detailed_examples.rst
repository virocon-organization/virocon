*****************
Detailed examples
*****************

.. warning::
    Stay tuned! We are currently working on this chapter.
    In the meantime if you have any questions feel free to open an issue.

This chapter will explain how the structure of the joint distribution model is created in virocon. The process of
estimating the parameter values of a joint distribution, the “fitting” is explained in more detail by means of two
examples. To create an environmental contour, first, we need to define a joint distribution. Then, we can choose a
specific contour method and initiate the calculation. virocon uses so-called global hierarchical models to define the
joint distribution and offers four common methods how an environmental contour can be defined based on a given joint
distribution. Generally, virocon provides two ways of creating a joint model and calculating a contour. The first option
is using an already predefined model, which was explained before in the quick start section. The second option is
defining a custom statistical model.

If the joint distribution is known, the procedure of calculating an environmental contour with virocon can be summarized as:

1.	Load the environmental data that should be described by the joint model.
2.	Define the dependence structure of the joint model that we will use to describe the environmental data. To define a joint model, we define the univariate parametric distributions and the dependence structure. The dependence structure is defined using parametric functions.
3.	Define the univariate parametric distributions.
    a.	Create a first, independent univariate distribution.
    b.	Create another, usually dependent univariate distribution and define its dependency on the previous distributions.
    c.	Repeat step 2, until you have created a univariate distribution for each environmental variable.
4.	Estimate the parameter values of the MultivariateModel (fitting).
5.	Define the contour’s return period and environmental state duration.
6.	Choose a type of contour: IFormContour, ISormContour, DirectSamplingContour or HighestDensityContour.
