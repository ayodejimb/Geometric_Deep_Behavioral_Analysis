function extract_skeleton_all(Project_Path)

% **** This function is for all ******

% Load the project_info structure
load([Project_Path,filesep,'Project_info']);

% Find Index of Videos with frames non_extracted
Idx2use = [];
Idx2use = find(Project.Project_List.is_frame == 1 & Project.Project_List.is_OF_coord == 1 ...
            & Project.Project_List.is_old_obj_coord == 1 & Project.Project_List.is_new_obj_coord == 1 ...
            & Project.Project_List.is_norm == 1 & Project.Project_List.is_norm_old_obj_coord ==1 & ...
            Project.Project_List.is_norm_new_obj_coord == 1 & Project.Project_List.is_distance == 0);
Idx2use = Idx2use';

%Loop normalization
for v = Idx2use

    % Set the start of the timer
    t_start = []; t_stop = [];
    t_start = tic;
    
    % First, let's import the object coordinates that we have saved
    load([Project.Path.new_obj_Coordinates, filesep, Project.Project_List.Video_List{v}]);
    new_obj_coords = new_obj_coordinates.Coord; 
    load([Project.Path.old_obj_Coordinates, filesep, Project.Project_List.Video_List{v}]);
    old_obj_coords = old_obj_coordinates.Coord;
    load([Project.Path.norm_new_obj_Coords, filesep, Project.Project_List.Video_List{v}]);
    norm_new_obj_coords = new_norm_coord;
    load([Project.Path.norm_old_obj_Coords, filesep, Project.Project_List.Video_List{v}]);
    norm_old_obj_coords = old_norm_coord;

    % Second, let's import the Image coordinates (in pixels) that we have saved
    load([Project.Path.image_coordinates, filesep, Project.Project_List.Video_List{v}]);
    image_coords = DLC_Data; 

    % Third, let's import the normalized coordinates that we have saved
    load([Project.Path.Coordinates, filesep, Project.Project_List.Video_List{v}]);
    norm_coords = Norm_DLC_output;

    varTypes = [];
    for i = 1:size(DLC_Data,2)
        varTypes{i} = 'double';
    end

    column_names = Norm_DLC_output.Properties.VariableNames;
    
    Sz = [1 size(column_names,2)];
    frames_table = table('Size', Sz, 'VariableTypes', varTypes, 'VariableNames', column_names);

    old_norm_coord_for_frame = [];
    new_norm_coord_for_frame = []; 

    for i = 1:size(norm_old_obj_coords,2)
        old_norm_coord_for_frame = [old_norm_coord_for_frame,norm_old_obj_coords(1,i),norm_old_obj_coords(2,i)];
        new_norm_coord_for_frame = [new_norm_coord_for_frame,norm_new_obj_coords(1,i),norm_new_obj_coords(2,i)];
    end   

    % Now, let's loop into the image coordinates frame by frame
    for i = 1:size(DLC_Data,1)
    
        frame_data = norm_coords{i,:}; % the normalized coord here

        frame_data = array2table(frame_data, 'VariableNames', column_names); % Convert to table to be able to add in the main one
        
        frames_table = [frames_table; frame_data];

    end

    % Initialize the progress bar
    fprintf([Project.Project_List.Video_List{v},' Distance Computation progress:   0%%']);
            

    % Calculate the current progress percentage
    progress = [];
    progress = (size(DLC_Data,2)) / (size(DLC_Data,2)) * 100;

    % Update the progress bar in the command window
    fprintf('\b\b\b\b%3d%%', round(progress));

    % Delete the junk first line of the table
    frames_table(1,:) = [];

    % Removing the first column (Numbering column) and the likelihoods
    % columns here. If you need the likelihood column in this table in the future,
    % comment the below and re-run this function
    frames_table(:, [1,4,7,10,13,16,19,22,25]) = [];

    % Update project
    Project.Project_List.is_distance(v) = 1;

    % Save the frames that satifies the distance condition as matlab file and csv
    save([Project.Path.frames_on_distance,filesep,Project.Project_List.Video_List{v}], 'frames_table', '-v7.3');
    writetable(frames_table, [Project.Path.frames_on_distance, filesep, Project.Project_List.Video_List{v},'.csv']);
    

    t_stop = toc(t_start);
    disp([' done in ', num2str(t_stop/60), ' min']);

    fprintf('\n'); % Print a newline to move to the next line after the loop

end

fclose('all');

% Save updated project
save([Project_Path,filesep,'Project_info'], 'Project', '-v7.3');