from pyboy import PyBoy
import keyboard
boy = PyBoy(rom_path='./roms/pokemon_pinball.gbc', window='sdl2', sound_emulated=False)
boy.set_emulation_speed(0)
boy.game_wrapper.start_game()
boy.set_emulation_speed(1)


count = 0
while True:
    count+= 1
    if count % 10 == 0:
        boy.button_press('A')
    boy.tick()

    print(f"{count}")
    print(f"Balls left: {boy.game_wrapper.balls_left}")
    print(f"Lost ball during saver: {boy.game_wrapper.lost_ball_during_saver}")


    # Wait for any key press
    keyboard.read_event(suppress=True)
    
    # Check if the pressed key is 'q' to exit
    if keyboard.is_pressed('q'):
        print("Exiting loop!")
        break