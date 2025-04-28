function get_arena_coordinates(Project_Path)

% This function gets the four corners of the open-field

% Load the project_info structure
load([Project_Path,filesep,'Project_info']);

% Find Index of Videos with frames non_extracted
Idx2use = [];
Idx2use = find(Project.Project_List.is_frame == 1 & Project.Project_List.is_OF_coord == 0);
Idx2use = Idx2use';

if isempty(Idx2use)
    disp('!!WARNING!!: Either frames were not extracted or arena coordinates already determined for this video, please verify the Project_List table')
end

% Loop frames coordinates
disp('Frame of the video will appear, click on the 4 corners of the Open Field following this specific order:')
disp('1- Top Left | 2- Top Right | 3- Bottom Right | 4- Bottom Left')

for v = Idx2use
    IMG = [];
    IMG = imread([Project.Path.Frames,filesep,Project.Project_List.Video_List{v},'.jpg']);
    fig1 = figure('WindowState','fullscreen'); 
    imshow(IMG);

    x_corner = []; y_corner = [];
    [x_corner, y_corner] = ginput(4);

    close(fig1);

    % Store Arena Corordinates
    Arena_Coordinates.Coord(1,:) = x_corner;
    Arena_Coordinates.Coord(2,:) = y_corner;
    Arena_Coordinates.Pts_Idx = {'Top_Left', 'Top_Right', 'Bottom_Right', 'Bottom_Left'};

    % Compute the different values for next_step normalization

    OA_OC = []; Theta = []; RM = []; Ref_Vector_X = []; norm_unit_X = []; Ref_Vector_Y = []; norm_unit_Y = [];

    % Vector from origin of OF and origin of camera
    OA_OC = [-Arena_Coordinates.Coord(1,1); -Arena_Coordinates.Coord(2,1)];

    % Angle between camera referential and OF referential
    Theta = -atan((Arena_Coordinates.Coord(2,2)-Arena_Coordinates.Coord(2,1))/(Arena_Coordinates.Coord(1,2)-Arena_Coordinates.Coord(1,1)));

    % Rotation matrix
    RM = [cos(Theta), -sin(Theta); sin(Theta), cos(Theta)];

    % Unit for normalize X axis: norm of the vector corresponding to the
    % Top side of the OF
    Ref_Vector_X = [(Arena_Coordinates.Coord(1,2)-Arena_Coordinates.Coord(1,1)); (Arena_Coordinates.Coord(2,2)-Arena_Coordinates.Coord(2,1))]; % Top side of the OF considered as reference for scale
    norm_unit_X = norm(Ref_Vector_X); % norm of the top side of the OF used to normalize all the distances

    % Unit for normalize Y axis: norm of the vector corresponding to the
    % Left side of the OF
    Ref_Vector_Y = [(Arena_Coordinates.Coord(1,4)-Arena_Coordinates.Coord(1,1)); (Arena_Coordinates.Coord(2,4)-Arena_Coordinates.Coord(2,1))]; % Top side of the OF considered as reference for scale
    norm_unit_Y = norm(Ref_Vector_Y); % norm of the top side of the OF used to normalize all the distances

    % Storing variables
    Arena_Coordinates.Norm_var = struct('OA_OC', OA_OC, 'Theta', Theta, 'RM', RM, 'norm_unit_X', norm_unit_X, 'norm_unit_Y', norm_unit_Y);

    % Update project
    Project.Project_List.is_OF_coord(v) = 1;

    % Save Arena coordinates
    save([Project.Path.Arena_Coordinates, filesep,Project.Project_List.Video_List{v}], 'Arena_Coordinates', '-v7.3');

end

% Save Project
save([Project_Path,filesep,'Project_info'], 'Project', '-v7.3');
