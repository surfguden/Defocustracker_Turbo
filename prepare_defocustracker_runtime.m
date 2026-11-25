function runtimeFolder=prepare_defocustracker_runtime
% Unpack and register the bundled Windows CUDA runtime once per package.
% Called by setup and by CUDA initialization. No GPU/toolkit is needed here.
runtimeFolder='';
if ~ispc||~strcmp(computer('arch'),'win64'),return,end
root=fileparts(mfilename('fullpath'));manifestPath=fullfile(root,'bin','manifest.json');
if ~isfile(manifestPath),return,end % Optional source-only/CPU-only installation.
manifest=jsondecode(fileread(manifestPath));package=manifest.runtime;
cudaFolder=fullfile(root,'bin','cuda');
runtimeFolder=fullfile(cudaFolder,'runtime',package.sha256);
marker=fullfile(runtimeFolder,'READY');
if ~cacheReady(runtimeFolder,marker,package)
    if ~isfolder(runtimeFolder),mkdir(runtimeFolder);end
    joinedZip=[tempname(fullfile(cudaFolder,'runtime')) '.zip'];
    cleanup=onCleanup(@()removeTemporaryFile(joinedZip));
    fprintf('Preparing bundled CUDA runtime (first use)...\n');
    joinParts(cudaFolder,package.parts,joinedZip);
    assert(strcmp(sha256File(joinedZip),package.sha256), ...
        'DefocusRuntime:ArchiveIntegrity','Bundled CUDA runtime archive is incomplete or damaged.');
    unzip(joinedZip,runtimeFolder);
    for k=1:numel(package.files)
        entry=package.files(k);path=fullfile(runtimeFolder,entry.name);
        assert(isfile(path)&&strcmp(sha256File(path),entry.sha256), ...
            'DefocusRuntime:LibraryIntegrity','Bundled runtime library is incomplete or damaged: %s',entry.name);
    end
    fid=fopen(marker,'w');assert(fid>=0,'DefocusRuntime:WriteFailed','Cannot write runtime cache marker.');
    close=onCleanup(@()fclose(fid));fprintf(fid,'%s',package.sha256);clear close cleanup
end
if ~any(strcmpi(strsplit(getenv('PATH'),pathsep),runtimeFolder))
    setenv('PATH',[runtimeFolder pathsep getenv('PATH')]);
end
end

function ready=cacheReady(folder,marker,package)
ready=isfile(marker);if ~ready,return,end
ready=strcmp(strtrim(fileread(marker)),package.sha256);if ~ready,return,end
for k=1:numel(package.files)
    entry=package.files(k);file=dir(fullfile(folder,entry.name));
    if isempty(file)||file(1).bytes~=entry.bytes,ready=false;return,end
end
end

function joinParts(folder,parts,destination)
output=fopen(destination,'wb');assert(output>=0,'DefocusRuntime:WriteFailed','Cannot create temporary runtime archive.');
outputCleanup=onCleanup(@()fclose(output));
for k=1:numel(parts)
    part=parts(k);path=fullfile(folder,part.name);info=dir(path);
    assert(~isempty(info)&&info(1).bytes==part.bytes, ...
        'DefocusRuntime:MissingPart','Missing/incomplete runtime package part: %s',part.name);
    input=fopen(path,'rb');assert(input>=0,'DefocusRuntime:ReadFailed','Cannot read runtime package part: %s',part.name);
    inputCleanup=onCleanup(@()fclose(input));
    while true
        chunk=fread(input,1048576,'*uint8');if isempty(chunk),break,end
        assert(fwrite(output,chunk,'uint8')==numel(chunk),'DefocusRuntime:WriteFailed','Cannot write temporary runtime archive.');
    end
    clear inputCleanup
end
end

function result=sha256File(path)
digest=java.security.MessageDigest.getInstance('SHA-256');
fid=fopen(path,'rb');assert(fid>=0,'DefocusRuntime:ReadFailed','Cannot read %s',path);
cleanup=onCleanup(@()fclose(fid));
while true
    chunk=fread(fid,1048576,'*uint8');if isempty(chunk),break,end
    digest.update(typecast(chunk,'int8'));
end
result=lower(reshape(dec2hex(typecast(digest.digest(),'uint8'),2).',1,[]));
end

function removeTemporaryFile(path)
if isfile(path),delete(path);end
end
