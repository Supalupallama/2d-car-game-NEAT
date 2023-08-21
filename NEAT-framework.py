import pygame as pg
import neat
import random
import os

# Initialize Pygame
pg.init()

# Set screen dimensions
WINDOWWIDTH = 1200
WINDOWHIGHT = 800
clock = pg.time.Clock()
FPS = 60

# Create screen object
screen = pg.display.set_mode((WINDOWWIDTH, WINDOWHIGHT))

# Set caption
pg.display.set_caption("NEAT Bot")


vel_multiply = 5 # increasing and rect speed fairly fast 

screen = pg.display.set_mode((WINDOWWIDTH, WINDOWHIGHT))
clock = pg.time.Clock()
FPS = 60
rectvel = 13
ballvel = pg.Vector2(random.choice([100, -100]), random.randint(-60, 60)) * vel_multiply
ballpos = pg.Vector2(WINDOWWIDTH/2, WINDOWHIGHT/2)

rect1pos_y = 20 
rect2pos_y = 20 



def draw():
    pg.draw.circle(screen, (255, 0, 0), ballpos, 20, 0)
    pg.draw.rect(screen, (0, 0, 255), (1870, rect2pos_y, 30, 280))
    pg.draw.rect(screen, (0, 0, 255), (20, rect1pos_y, 30, 280))


def draw(screen, clock):
    font = pg.font.Font(None, 30)
    fps = str(int(clock.get_fps()))
    fps_text = font.render(fps, 1, pg.Color("coral"))
    screen.blit(fps_text, (WINDOWWIDTH - 30, 0))

    score_label = font.render("Gens: " + str(generation-1),1,(255,255,255))
    screen.blit(score_label, (10, 10))


# Set up game variables
generation = 0

def eval_genomes(genomes, config):
    global generation
    generation += 1
    total_succes = 0

    # Loop through each genome in the current generation
    for genome_id, genome in genomes:
        # Set up NEAT neural network
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        # Set initial fitness to 0
        genome.fitness = 0

        # Game loop
        running = True
        while running:
            clock.tick(FPS) # comment out to unlock FPS
            # Handle events
            for event in pg.event.get():
                if event.type == pg.QUIT or event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                    running = False
                    quit()
            num1 = random.random() # random float 
            
            # Get inputs for neural network
            inputs = [num1]

            # Get output from neural network
            output = net.activate(inputs)
            output_value = output[0]

            # Use output to control game character
            print("input: " + str(num1) + " output: " + str(output_value) + "total succes:  " + str(total_succes))
            # Update fitness based on game performance
            if num1 - 0.02 <= output_value <= num1 + 0.02:
                genome.fitness = 10
                total_succes += 1
                print("SUCCES")
            else:
                genome.fitness -= 10
                print("FAILURE")
                running = False

            if total_succes >= 2000:
                running = False
            
            genome.fitness = (1 / abs(num1 - output_value))
            

            # Update screen
            screen.fill((0, 0, 0))
            draw(screen, clock)
            pg.display.update()

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

    # Run NEAT algorithm for up to 50 generations
    winner = p.run(eval_genomes, 50)

if __name__ == "__main__":
    # Set path to NEAT configuration file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run(config_path)