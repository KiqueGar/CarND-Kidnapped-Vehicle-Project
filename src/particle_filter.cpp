/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	for(int i=0; i<num_particles; i++){
		//Create particle
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		//Append to particles
		particles.push_back(p);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	normal_distribution<double> noise_x(0, std_pos[0]);
	normal_distribution<double> noise_y(0, std_pos[1]);
	normal_distribution<double> noise_theta(0, std_pos[2]);
	for(int i = 0; i< num_particles; i++){

		if(fabs(yaw_rate)>0.001){
			float VelOverYaw = velocity/ yaw_rate;
			float arc = yaw_rate*delta_t;
			particles[i].x += VelOverYaw*(sin(particles[i].theta +arc)-sin(particles[i].theta));
			particles[i].y += VelOverYaw*(cos(particles[i].theta)+cos(particles[i].theta + arc));
			particles[i].theta += arc;
			//NOMRALIZE THETA

		}
		else{
			particles[i].x = velocity*delta_t*cos(particles[i].theta);
			particles[i].y = velocity*delta_t*sin(particles[i].theta);
		}
		//Add noise
		particles[i].x += noise_x(gen);
		particles[i].y += noise_y(gen);
		particles[i].theta +=noise_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i = 0; i<observations.size(); i++){
		//Current observation (absolute coordinates)
		LandmarkObs obs = observations[i];
		//Min distance initialized as maximum (~100k)
		float min_distance = 99999;
		//Id of best observation (-1 is none)
		int map_id = -1;
		//Run through predictions
		for (int j=0; j< predicted.size(); j++){
			LandmarkObs pred = predicted[j];
			//Current distance current to predicted
			double cur_dist = dist(obs.x, obs.y, pred.x, pred.y);
			//Replace id for best estimate landmark
			if (cur_dist < min_distance){
				min_distance= cur_dist;
				map_id= pred.id;
			}
		}
		//Set the best estimete as observation
		observations[i].id = map_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	//For each particle
	for (int i =0; i< num_particles; i++){
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;
		//This will hold predicted landmark locations
		vector<LandmarkObs> predictions;
		//For each landmark
		for(int j=0; j< map_landmarks.landmark_list.size(); j++){
			//get position and ID
			float l_x = map_landmarks.landmark_list[j].x_f;
			float l_y = map_landmarks.landmark_list[j].x_f;
			float l_id = map_landmarks.landmark_list[j].id_i;
			//Discriminate for only within sensor range
			if (fabs(l_x-p_x)<=sensor_range && fabs(l_y - p_y)<= sensor_range){
				//Add to predictions
				predictions.push_back(LandmarkObs{l_id,l_x,l_y});
			}
		}
		//Transform observations
		std::vector<LandmarkObs> transformed_obs;
		for (int j =0; j< observations.size(); j++){ 
			//T = Rotation*Observation + particle
			//T = [cos()	sin()]*[Ox] + [px]
			//		[sin()	cos()] [Oy]	  [py]
			double t_x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x;
			double t_y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y;
			transformed_obs.push_back(LandmarkObs{observations[j].id, t_x, t_y});
		}
		//Associate observation to landmark
		dataAssociation(predictions, transformed_obs);

		for (int j = 0; j< transformed_obs.size(); j++){
			double o_x, o_y, pred_x, pred_y;
			o_x = transformed_obs[j].x;
			o_y = transformed_obs[j].y;
			int associated_pred = transformed_obs[j].id;

			// Get prediction coordinates (A better way for no iterate?)
			for(int k =0; k<predictions.size(); k++){
				if(predictions[k].id== associated_pred){
					pred_x = predictions[k].x;
					pred_y = predictions[k].y;
				}
			}

			//Weight observtion
			double s_x, s_y;
			s_x = std_landmark[0];
			s_y = std_landmark[1];
			double normalizer;
			normalizer = 1/(2*M_PI*s_x*s_y);
			double obs_w =normalizer*exp( -( pow(pred_x-o_x,2)/(2*pow(s_x,2)) + (pow(pred_y - o_y, 2)/(2*pow(s_y,2)))));
			particles[i].weight= obs_w;
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution


}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
