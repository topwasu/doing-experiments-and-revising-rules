#!/bin/bash

python run.py agent/proposal=particle_filter agent.proposal.num_max_calls_per_it=5 agent.proposal.num_particles=25 agent.theta_mean=0.70 agent.delta_mean=0.90 agent.n_neighbors=7 database_path=completions_v3.db task_number=8 seed=4
