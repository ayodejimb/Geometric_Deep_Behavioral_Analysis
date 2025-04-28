function normalize_data(Project_Path)

% This function will normalize the skeleton data based on the open-field
% coordinates

% Load the project_info structure
load([Project_Path,filesep,'Project_info']);

% Find Index of Videos with frames non_extracted
Idx2use = [];
Idx2use = find(Project.Project_List.is_frame == 1 & Project.Project_List.is_OF_coord == 1 ...
            & Project.Project_List.is_old_obj_coord == 1 & Project.Project_List.is_new_obj_coord == 1 & Project.Project_List.is_norm == 0);
Idx2use = Idx2use';

if isempty(Idx2use)
    disp('!!WARNING!!: Either frames were not extracted, arena coordinates not determined or skeleton data already normalized for this video, please verify the Project_List table')
end

%Loop normalization
for v = Idx2use

    % Set the start of the timer
    t_start = []; t_stop = [];
    t_start = tic;

    DLC_file_info = [];

    DLC_file_info = dir(fullfile(Project.Path.DLC_output, [Project.Project_List.Video_List{v}, '*.csv']));

    DLC_file_name = [];
    DLC_file_name = DLC_file_info.name;

    textformat = [];
    for i = 1:(numel(Project.Bodyparts)*3+1)
        textformat = [textformat, '%s'];
    end
    
    pathh = fullfile(Project.Path.DLC_output, [DLC_file_name]);
    pathh = strrep(pathh, '\', '/');
    fid = fopen(pathh);

    C = textscan(fid, '%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s', 'Delimiter', ',');

    % Creating cell vector of variable names
    Var = [];
    for c = 1:numel(C)
        Var{c} = [C{1,c}{2},'_', C{1,c}{3}];
    end

    Var{1} = 'Frames';

    DLC_Data = readtable(pathh, 'NumHeaderLines',3,'ReadVariableNames',false );

    Old_vars = [];
    for i = 1:size(DLC_Data, 2)
        Old_vars{i} = ['Var',num2str(i)];
    end

    % Replace Variable title of skeleton_Data
    DLC_Data = renamevars(DLC_Data,Old_vars,Var);

    % Load the coordinates from arena_coordinates
    load([Project.Path.Arena_Coordinates, filesep, Project.Project_List.Video_List{v}]);

    % Create the Table that will have the normalized data
    Sz = [1 size(DLC_Data,2)];
    varTypes = [];
    for i = 1:size(DLC_Data,2)
        varTypes{i} = 'double';
    end
    Norm_DLC_output = table('Size', Sz, 'VariableTypes', varTypes, 'VariableNames',Var);

    % Initialize the progress bar
    fprintf([Project.Project_List.Video_List{v},' Normalization progress:   0%%']);

    for t = 1:size(DLC_Data,1)
        for b = 1:numel(Project.Bodyparts)
            BodyP_coord(:,b) = [DLC_Data.([Project.Bodyparts{b},'_x'])(t); DLC_Data.([Project.Bodyparts{b},'_y'])(t)];
        end

        % First we realign the coordinate from the camera referential to
        % the Open Field Referential
        Realign_coord = []; New_origin = [];
        New_origin = BodyP_coord + Arena_Coordinates.Norm_var.OA_OC;
        Realign_coord = Arena_Coordinates.Norm_var.RM*New_origin;

        Norm_coord = [];

        % Normalize data taking X & Y length in account
        Norm_coord(1,:) = Realign_coord(1,:)./Arena_Coordinates.Norm_var.norm_unit_X;
        Norm_coord(2,:) = Realign_coord(2,:)./Arena_Coordinates.Norm_var.norm_unit_Y;


        Temp_Table = table('Size', Sz, 'VariableTypes', varTypes, 'VariableNames',Var);
        Temp_Table.(Var{1}) = t;

        for b = 1:numel(Project.Bodyparts)
            Temp_Table.([Project.Bodyparts{b},'_x']) = Norm_coord(1,b);
            Temp_Table.([Project.Bodyparts{b},'_y']) = Norm_coord(2,b);
        end
        Norm_DLC_output = [Norm_DLC_output; Temp_Table];

        clear Temp_Table;

        % Calculate the current progress percentage
        progress = [];
        progress = t / (size(DLC_Data,1)) * 100;

        % Update the progress bar in the command window
        fprintf('\b\b\b\b%3d%%', round(progress));
    end

    % Delete the junk first line of the table
    Norm_DLC_output(1,:) = [];

    % Add the likelihood data from the old table to the new one
    for b = 1:numel(Project.Bodyparts)
        Norm_DLC_output.([(Project.Bodyparts{b}),'_likelihood']) = DLC_Data.([(Project.Bodyparts{b}),'_likelihood']);
    end

    % Save the new table in .mat and .csv
    save([Project.Path.Coordinates, filesep, Project.Project_List.Video_List{v}], 'Norm_DLC_output', '-v7.3');

    writetable(Norm_DLC_output, [Project.Path.Coordinates, filesep, Project.Project_List.Video_List{v},'.csv']);

    % Save the Image coordiantes in .mat and .csv
    save([Project.Path.image_coordinates, filesep, Project.Project_List.Video_List{v}], 'DLC_Data', '-v7.3');

    writetable(DLC_Data, [Project.Path.image_coordinates, filesep, Project.Project_List.Video_List{v},'.csv']);


    % Update project
    Project.Project_List.is_norm(v) = 1;

    t_stop = toc(t_start);
    disp([' done in ', num2str(t_stop/60), ' min']);

    fprintf('\n'); % Print a newline to move to the next line after the loop
end

% Save updated project
save([Project_Path,filesep,'Project_info'], 'Project', '-v7.3');