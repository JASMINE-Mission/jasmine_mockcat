# The JASMINE mock catalogue of the Galactic centre in NIR

This code calculates the Bayesian posterior distribution function of the distance and component (thin disc, thick disc, bar, NSD or NSC) using the available data (limited to J,H,Ks and/or parallax for the moment) and the Koshimoto E+E_X model 5 from Koshimoto et al. 2021 as prior.

In the process of computing the probabilities of each star, it also samples them to generate 1 mock particle that resembles the observed star as much as possible. The mock particle has mock photometry, distance and velocities, from which we can generate magnitudes, colours, parallaxes, proper motions and radial velocities.

To-Do:
- Store all relevant information in the name file of the posterior PDF: source_id, l, b, magnitudes, parallax (pmra,pmdec,vlos)
- Update kinematic model of the bar (currently, it does not have a quadrupole) and of the NSC. For the bar, we should use the Portail+17 M2M model. For the NSC, we should use the two-components distribution function of Chatzopoulos+15. In both cases, we should use the first and second order moments like we do now for the NSD.
- Add the measured proper motions and radial velocities to the Distance posterior.
- Add compatibility with .fits files
