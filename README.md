# The JASMINE mock catalogue of the Galactic centre in NIR

This code calculates the Bayesian posterior distribution function of the distance and component (thin disc, thick disc, bar, NSD or NSC) using the available data (limited to J,H,Ks and/or parallax for the moment) and the Koshimoto E+E_X model 5 from Koshimoto et al. 2021 as prior.

In the process of computing the probabilities of each star, it also samples them to generate 1 mock particle that resembles the observed star as much as possible. The mock particle has mock photometry, distance and velocities, from which we can generate magnitudes, colours, parallaxes, proper motions and radial velocities.

To-Do:
- Add the measured proper motions and radial velocities to the Distance posterior.
- Code it in a way so that it can process many line of sights autonomously. 
