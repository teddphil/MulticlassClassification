import platform
import matplotlib

if platform.system() == 'Darwin':
    font_display = 'Helvetica Neue'
elif platform.system() == 'Linux':
    font_display = 'Nimbus Sans'
else:
    font_display = 'Tahoma'

# General Colors
color_profile_general = {
    'blue':     '#2A6EBB',
    'red':      '#CD202C',
    'yellow':   '#F0AB00',
    'purple':   '#7D5CC6',
    'aqua':     '#00B2A9',
    'pink':     '#C50084',
    'green':    '#69BE28',
    'blue_light':    '#B4CFEE',
    'red_light':     '#F4B6BA',
    'yellow_light':  '#FFE8AE',
    'purple_light':  '#ECE8F7',
    'aqua_light':    '#70FFF8',
    'pink_light':    '#FF83D6',
    'green_light':   '#CDEFB4'
}

# Figure Sizes (Renamed to snake_case)
fig_size       = (4, 3)
fig_size_2      = (3.5, 3.5)
fig_size_5      = (5, 5)
fig_size_large  = (8, 8)
font_size_2     = 16
font_size_large = 18
font_size_caption = 14

# Figure Adjustments (Renamed to snake_case)
fig_adjust_b = 0.17
fig_adjust_l = 0.21
fig_adjust_r = 0.91
fig_adjust_t = 0.87
