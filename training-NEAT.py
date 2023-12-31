import pygame as pg
import math
import neat
import os
import pickle



pg.init()
# Set screen dimensions
WINDOWWIDTH = 1920
WINDOWHIGHT = 1080
clock = pg.time.Clock()
FPS = 60

# Set caption
pg.display.set_caption("NEAT Bot")

screen = pg.display.set_mode((WINDOWWIDTH, WINDOWHIGHT))
clock = pg.time.Clock()

#load image in folder "images"
IMG = pg.transform.scale(pg.image.load("images/car.png"), (35, 67)).convert_alpha()
IMG_map = pg.transform.scale(pg.image.load("images/map.png"), (1920, 1080)).convert()
BORDER = pg.transform.scale(pg.image.load("images/border.png"), (1920, 1080)).convert_alpha()
BORDER_MASK = pg.mask.from_surface(BORDER)

generation = 0

class Car: 
	def __init__(self):
		self.pos = pg.Vector2(1000, 320)
		self.vel = pg.Vector2(0, 0)
		self.turn_vel = 0
		self.angle = 90
		self.speed = 0
		self.rotated_rect = IMG.get_rect()
		self.rotated_rect.center = self.pos
		# Rotate the image around the new Rect object
		self.rotated_image = pg.transform.rotate(IMG, self.angle) 
		# Set the position of the rotated image to the position of the original image
		self.rotated_rect = self.rotated_image.get_rect(center=self.rotated_rect.center)
		self.raycasts = [] # List For Sensors / raycasts
		self.poi = None
		self.output = [0, 0, 0, 0]
		self.distance = 0

	def move(self):
		pressed_keys = pg.key.get_pressed()
		self.speed -= 0.03

		self.speed -= abs(self.output[0]) / 10
		
		
		if pressed_keys[pg.K_LEFT] and self.speed != 0 or self.output[1] > 0 and self.speed != 0:
			self.turn_vel += (0.53 * abs(self.speed)) / abs(self.speed)
		
		if pressed_keys[pg.K_RIGHT] and self.speed != 0 or self.output[1] < 0 and self.speed != 0:
			self.turn_vel -= (0.53 * abs(self.speed)) / abs(self.speed)

		self.speed /= 1.007  # friction
		self.turn_vel /= 1.175  # friction

		self.vel.x = math.cos(math.radians(self.angle - 90)) * self.speed
		self.vel.y = -math.sin(math.radians(self.angle - 90)) * self.speed
		
		self.pos += self.vel
		self.angle += self.turn_vel
		self.angle %= 360

		self.rotated_rect = IMG.get_rect()
		self.rotated_rect.center = self.pos
		# Rotate the image around the new Rect object
		self.rotated_image = pg.transform.rotate(IMG, self.angle) 
		# Set the position of the rotated image to the position of the original image
		self.rotated_rect = self.rotated_image.get_rect(center = self.rotated_rect.center)

		self.distance += abs(self.speed)

	def draw(self, screen, clock):
		screen.blit(IMG_map,(0, 0))
		font = pg.font.Font(None, 30)
		score_label = font.render("Generation: " + str(generation-1),1,(255,255,255))
		screen.blit(score_label, (10, 10))
		font = pg.font.Font(None, 30)
		fps = str(int(clock.get_fps()))
		fps_text = font.render(fps, 1, pg.Color("coral"))
		screen.blit(fps_text, (WINDOWWIDTH - 30, 0))
		screen.blit(self.rotated_image, self.rotated_rect) 


	def collision(self):
		car_mask = pg.mask.from_surface(self.rotated_image)
		self.poi = BORDER_MASK.overlap(car_mask, (self.rotated_rect[0], self.rotated_rect[1]))
		return self.poi
	

	def draw_raycast(self, screen):
		# Optionally Draw All raycasts
		for raycast in self.raycasts:
			position = raycast[0]
			pg.draw.line(screen, (0, 255, 0), (self.pos.x, self.pos.y), position, 1)
			pg.draw.circle(screen, (0, 255, 0), position, 5)
			

	def check_raycast(self, degree):
		length = 0
		x = self.pos.x
		y = self.pos.y
	
		if x <= WINDOWWIDTH - 3 and y <= WINDOWHIGHT - 3 and x > 0 and y > 0:
			while BORDER_MASK.get_at((x, y)) == 0 and length <= 500:
				length = length + 1
				x = int(self.pos.x + math.cos(math.radians(360 - (self.angle + degree))) * length)
				y = int(self.pos.y + math.sin(math.radians(360 - (self.angle + degree))) * length)
				

		# Calculate Distance To Border And Append To raycasts List
		dist = int(math.sqrt(math.pow(x - self.pos.x, 2) + math.pow(y - self.pos.y, 2)))
		self.raycasts.append([(x, y), dist])

	def get_data(self):
		return_values = [0] * len(self.raycasts)
		for i, raycast in enumerate(self.raycasts):
			return_values[i] = int(raycast[1] / 50)

		return return_values
	

def eval_genomes(genomes, config):
	view = False
	global generation
	generation += 1
	

	for genome_id, genome in genomes:
		car = Car()
		timer = 0
		# Set up NEAT neural network
		net = neat.nn.FeedForwardNetwork.create(genome, config)
		# Set initial fitness to 0
		genome.fitness = 0
		running = True

		while running:
			for event in pg.event.get():
				if event.type == pg.QUIT or event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
					running = False
					quit()
				if event.type == pg.KEYDOWN and event.key == pg.K_r:
					car = Car()

				if event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
					view = not view
					print(view)

			if car.collision() != None:
				break

			
			if car.vel.x == 0:
				timer += 200

			timer += 1
			if timer > 3000: # after certain time, enough for 2 laps about 30 seconds
				break	
			
			car.move()
			angles_to_check = [0, 50, 70, 83, 90, 97, 110, 130, 180]



			for d in angles_to_check:
				car.check_raycast(d)

			car.output = net.activate(car.get_data())
			

			genome.fitness += car.distance / 10000

			genome.fitness += car.output[0] / 50



			
			if view:
				screen.fill((0, 0, 0)) 
				clock.tick(FPS)
				car.draw(screen, clock)
				car.draw_raycast(screen)
				print(car.output)

				pg.display.update()
			car.raycasts.clear()
		
		



def run(config_file):
	# Load NEAT configuration
	config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
								neat.DefaultSpeciesSet, neat.DefaultStagnation,
								config_file)

	# Create population
	p = neat.Population(config)

	# Add reporter to show progress in terminal
	p.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	p.add_reporter(stats)

	# Run NEAT algorithm for up to 2000 generations
	num_generations = 5000
	
	for i in range(num_generations):
		# Run NEAT algorithm for 1 generation
		winner = p.run(eval_genomes, 1)

		# Save the winner genome using pickle
		with open(f'winner_genome_{i}.pkl', 'wb') as f:
			pickle.dump(winner, f)

if __name__ == "__main__":
	# Set path to NEAT configuration file
	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, "config.txt")
	run(config_path)
