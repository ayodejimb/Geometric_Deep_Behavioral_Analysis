function extract_video_frames(Project_Path)

% This function will extract one frame from each video
% This frames will be used to set the coordinates for the normalization
%
% INPUT:
%   - Project_Path:

% Load the project_info structure
load([Project_Path,filesep,'Project_info']);

% Find Index of Videos with frames non_extracted
Idx2use = [];
Idx2use = find(Project.Project_List.is_frame == 0);
Idx2use = Idx2use';

% Extract one frame and save it for each video
for v = Idx2use
    Vid = []; Frame = [];
  
    Vid = VideoReader(['video_all',filesep,(Project.Project_List.Video_List{v}),Project.Video_format]);
       
    Frame = read(Vid, randi([1000 4000],1));

    imwrite(Frame, [Project.Path.Frames,filesep,Project.Project_List.Video_List{v},'.jpg']);
    Project.Project_List.is_frame(v) = 1;
end

% save actualised project
save([Project_Path,filesep,'Project_info'], 'Project', '-v7.3');