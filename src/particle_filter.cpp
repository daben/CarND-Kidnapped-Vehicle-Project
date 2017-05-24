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

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  
  // Initialize the random generator.
  // Here I prefer the Mersenne Twister to the default random generator, its randomness quality is better guaranteed.
  // See http://en.cppreference.com/w/cpp/numeric/random/mersenne_twister_engine for more information.
  std::random_device seed;
  std::mt19937 random_generator(seed());
  
  // Normal distributions centered in the initial GPS estimates
  // using the GPS uncertainties as deviations.
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_yaw(theta, std[2]);
  
  // Set the number of particles - There is a tradeoff between speed and
  // accuracy. But this filter already works with 2 particles.
  num_particles = 10;
  
  // Initialize particle container
  particles.resize(num_particles);
  
  // Now initialize all particles randomly to their first position
  for(int i = 0; i < num_particles; ++i) {
    auto & particle = particles[i];
    particle.id = i;
    particle.x = dist_x(random_generator);
    particle.y = dist_y(random_generator);
    particle.theta = dist_yaw(random_generator);
    particle.weight = 1.0;
  }
  
  // Set initial weights all to 1
  weights.resize(num_particles, 1.0);
  
  // Initialization completed
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// DONE: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  // Handle the case without no movement
  const bool moving = fabs(velocity) > 1E-3;
  
  if (moving)
  {
    // Predict the state for the next time step using the bicycle model
    if (fabs(yaw_rate) < 1E-6)
    {
      // move with constant heading
      for(auto & particle : particles) {
        particle.x += velocity * delta_t * cos(particle.theta);
        particle.y += velocity * delta_t * sin(particle.theta);
      }
    } else {
      // move changing heading
      for(auto & particle : particles) {
        const double
          theta_0 = particle.theta,
          theta_t = theta_0 + yaw_rate * delta_t;
        
        particle.x += velocity / yaw_rate * (sin(theta_t) - sin(theta_0));
        particle.y -= velocity / yaw_rate * (cos(theta_t) - cos(theta_0));
        particle.theta += yaw_rate * delta_t;
      }
    }
  }

  // Add noise in any case
  
  // Initialize the random generator.
  std::random_device seed;
  std::mt19937 random_generator(seed());
  
  // Noise distributions
  std::normal_distribution<double> noise_x(0, std_pos[0]);
  std::normal_distribution<double> noise_y(0, std_pos[1]);
  std::normal_distribution<double> noise_yaw(0, std_pos[2]);
  
  for(auto & particle : particles) {
    particle.x += noise_x(random_generator);
    particle.y += noise_y(random_generator);
    particle.theta += noise_yaw(random_generator);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// DONE: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  for(auto & obs : observations) {
    // Find the closest prediction
    int closest_index = -1;
    double closest_distance = INFINITY;
    for(int i = 0; i < predicted.size(); ++i) {
      const double distance = dist(predicted[i].x, predicted[i].y, obs.x, obs.y);
      if (distance < closest_distance) {
        closest_distance = distance;
        closest_index = i; // index in predicted vector
      }
    }
    // Associate the observation to the landmark
    obs.id = closest_index;
  }
}

// Bivariate Normal Distribution PDF with diagonal covariance.
inline double norm2pdf (double x, double y, double mean_x, double mean_y, double std_x, double std_y) {
  double zx = (x - mean_x) / std_x;
  double zy = (y - mean_y) / std_y;
  return 0.5 * M_1_PI * exp(-0.5 * (zx * zx + zy * zy)) / (std_x * std_y);
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// DONE: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
  
  // Allocate once and reuse for each particle
  std::vector<LandmarkObs> predicted(observations.size());
  std::vector<LandmarkObs> p_landmarks;
  // Reserve a reasonable ammount
  p_landmarks.reserve(observations.size() * 2);

  for (auto & p : particles) {
    // Precompute trigonometry
    const double
      cos_theta = cos(p.theta),
      sin_theta = sin(p.theta);
    
    // Transform observations to the frame reference in the particle
    for (int i = 0; i < observations.size(); ++i) {
      const auto & obs = observations[i];
      predicted[i].x = p.x + obs.x * cos_theta - obs.y * sin_theta;
      predicted[i].y = p.y + obs.x * sin_theta + obs.y * cos_theta;
    }
    
    // Filter landmarks inside the sensor range
    p_landmarks.clear(); // (first reset the container)
    for (const auto & landmark : map_landmarks.landmark_list) {
      if (dist(landmark.x_f, landmark.y_f, p.x, p.y) <= sensor_range) {
        p_landmarks.push_back({landmark.id_i, (double)landmark.x_f, (double)landmark.y_f});
      }
    }
    
    if (!p_landmarks.empty()) {
      // Associate landmarks with observations
      dataAssociation(predicted, p_landmarks);
      // Update the weight of this particle
      double weight = 1.0;
      for(const auto & landmark : p_landmarks) {
        // map landmark to observation x, y
        double x = predicted[landmark.id].x;
        double y = predicted[landmark.id].y;
        // update weight with the bivariate gaussian probability.
        weight *= norm2pdf(x, y, landmark.x, landmark.y, std_landmark[0], std_landmark[1]);
      }
      // Update the particle weight
      p.weight = weight;
    } else {
      // Case where there are no landmarks in range... Ignore all observations.
      p.weight = 0.0;
    }
  }

  // Copy all the particle weights. It's better to iterate again on
  // all particles than to mess the previous loop.
  for(int i = 0; i < particles.size(); i++)
    weights[i] = particles[i].weight;
}

void ParticleFilter::resample() {
	// DONE: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  
  std::random_device seed;
  std::mt19937 random_generator(seed());
  // sample particles based on their weight
  std::discrete_distribution<> sample(weights.begin(), weights.end());

  // This vector could actually be allocated once...
  std::vector<Particle> new_particles(num_particles);
  for(auto & p : new_particles)
    p = particles[sample(random_generator)];
  particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
