function get_obj_2_coordinates(Project_Path)

% This function will get the boundaries of the second object

% Load the project_info structure
load([Project_Path,filesep,'Project_info']);

% Find Index of Videos with frames non_extracted
Idx2use = [];
Idx2use = find(Project.Project_List.is_frame == 1 & Project.Project_List.is_OF_coord == 1 ...
            & Project.Project_List.is_old_obj_coord == 1 & Project.Project_List.is_new_obj_coord == 0);
Idx2use = Idx2use';

if isempty(Idx2use)
    disp('!!WARNING!!: Either frames were not extracted or new object coordinates already determined for this video, please verify the Project_List table')
end

for v = Idx2use
    IMG = [];
    IMG = imread([Project.Path.Frames,filesep,Project.Project_List.Video_List{v},'.jpg']);
    fig1 = figure('WindowState','fullscreen'); 
    imshow(IMG);

    fontSize = 8;
    imshow(IMG, []);
    axis on;
    title('9 points Boundaries on New Obj', 'FontSize', fontSize);
    set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
    message = sprintf('This is for the new object. Click in that usual order!!!!');
    uiwait(msgbox(message));
    
    [x_corner, y_corner] = ginput(9);

    close(fig1);

    % Store object coordinates
    new_obj_coordinates.Coord(1,:) = x_corner;
    new_obj_coordinates.Coord(2,:) = y_corner;

    % Update project
    Project.Project_List.is_new_obj_coord(v) = 1;

    % Save the object's coordinates
    save([Project.Path.new_obj_Coordinates, filesep,Project.Project_List.Video_List{v}], 'new_obj_coordinates', '-v7.3');

end

% Save Project
save([Project_Path,filesep,'Project_info'], 'Project', '-v7.3');
