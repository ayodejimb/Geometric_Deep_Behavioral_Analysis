function create_project(Project_Path, DLC_output_Path, Video_Path, Bodyparts)

% INPUTS:
%   - Project_Path: Path where you want to store results
%   - DLC_output_path: Path of DeepLabCut csv files
%   - Video_Path: Path for videos
%   - Bodyparts: Cell Vector "{}" of the differents Bodyparts

% Create project folder
mkdir(Project_Path);

mkdir([Project_Path, filesep, 'Video_frames']);
mkdir([Project_Path, filesep, 'Norm_coordinates']);
mkdir([Project_Path, filesep, 'Arena_Coordinates']);
mkdir([Project_Path, filesep, 'Image_coordinates']);
mkdir([Project_Path, filesep, 'old_obj_Coordinates']);
mkdir([Project_Path, filesep, 'norm_old_obj_Coords']);
mkdir([Project_Path, filesep, 'new_obj_Coordinates']);
mkdir([Project_Path, filesep, 'norm_new_obj_Coords']);
mkdir([Project_Path, filesep, 'frames_on_distance']);
mkdir([Project_Path, filesep, 'exploration_times']);

% Create the list of video names for looping all others function
List_File = dir(Video_Path);
List_File = List_File(~cellfun(@(x) x==1, {List_File.isdir})); % Remove line non corresponding to files
List_File = List_File(~cellfun(@(x) strcmp(x, '.DS_Store'), {List_File.name})); % Remove DS_store cache file if running on mac

% Create a table that will have the name of video and the status to adapt
% for when need to add videos
Video_List = {};
for f = 1:length(List_File)
    [~, name, ~] = fileparts(List_File(f).name);
    Video_List{f,1} = name;
end

Video_format = [];
[~, ~, Video_format] = fileparts(List_File(1).name); 

is_frame = zeros(length(Video_List),1);
is_OF_coord = zeros(length(Video_List),1);
is_old_obj_coord = zeros(length(Video_List),1);
is_new_obj_coord = zeros(length(Video_List),1);
is_norm_old_obj_coord = zeros(length(Video_List),1);
is_norm_new_obj_coord = zeros(length(Video_List),1);
is_norm = zeros(length(Video_List),1);
is_distance = zeros(length(Video_List),1);
is_extracted_frames = zeros(length(Video_List),1);

Project_List = [];
Project_List = table(Video_List, is_frame, is_OF_coord, is_old_obj_coord, is_new_obj_coord, is_norm,is_norm_old_obj_coord ,is_norm_new_obj_coord, is_distance, is_extracted_frames);
% Create an Project structure that will contain all informations for
% next functions

Project.Path.Project = Project_Path;
Project.Path.Video = Video_Path;
Project.Path.DLC_output = DLC_output_Path;
Project.Path.Frames = [Project_Path, filesep, 'Video_frames'];
Project.Path.Coordinates = [Project_Path, filesep, 'Norm_coordinates'];
Project.Path.Arena_Coordinates = [Project_Path, filesep, 'Arena_Coordinates'];
Project.Path.image_coordinates = [Project_Path, filesep, 'Image_coordinates'];
Project.Path.frames_on_distance = [Project_Path, filesep, 'frames_on_distance'];
Project.Path.old_obj_Coordinates = [Project_Path, filesep, 'old_obj_Coordinates'];
Project.Path.new_obj_Coordinates = [Project_Path, filesep, 'new_obj_Coordinates'];
Project.Path.norm_old_obj_Coords = [Project_Path, filesep, 'norm_old_obj_Coords'];
Project.Path.norm_new_obj_Coords = [Project_Path, filesep, 'norm_new_obj_Coords'];
Project.Path.exploration_times = [Project_Path, filesep, 'exploration_times'];

Project.Project_List = Project_List;
Project.Creation_Date = datetime('now', 'Format','dd-MMM-yyyy');
Project.Bodyparts = Bodyparts;
Project.Video_format = Video_format;

% Save the structure
save([Project_Path,filesep,'Project_info'], 'Project', '-v7.3');