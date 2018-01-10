--[[
Copyright 2014 Google Inc. All Rights Reserved.

Use of this source code is governed by a BSD-style
license that can be found in the LICENSE file or at
https://developers.google.com/open-source/licenses/bsd
]]

local mnist_cluttered = require 'mnist_cluttered'
local min_distractors = 8
local max_distractors = 8 -- for batch generation
local megapatch_w = 40
local train_size = 200000
local val_size = 100000
local test_size = 100000

print("train.t7...")
for num_dists = min_distractors,max_distractors,1 do
    local dataConfig = {megapatch_w=megapatch_w, num_dist=num_dists, datasetPath='mnist/train.t7'}
    local dataInfo = mnist_cluttered.createData(dataConfig)
    local data = {}
    local targets = {}
    for i = 1,train_size do
        local observation, target = unpack(dataInfo.nextExample())
        table.insert(data, observation:clone())
        table.insert(targets, target:clone())
    end
    torch.save('th-mnist/train_'..num_dists..'.t7', {data, targets})
end
print("valid.t7...")
for num_dists = min_distractors,max_distractors,1 do
    local dataConfig = {megapatch_w=megapatch_w, num_dist=num_dists, datasetPath='mnist/valid.t7'}
    local dataInfo = mnist_cluttered.createData(dataConfig)
    local data = {}
    local targets = {}
    for i = 1,val_size do
        local observation, target = unpack(dataInfo.nextExample())
        table.insert(data, observation:clone())
        table.insert(targets, target:clone())
    end
    torch.save('th-mnist/val_'..num_dists..'.t7', {data, targets})
end
print("test.t7...")
for num_dists = min_distractors,max_distractors,1 do
    local dataConfig = {megapatch_w=megapatch_w, num_dist=num_dists, datasetPath='mnist/test.t7'}
    local dataInfo = mnist_cluttered.createData(dataConfig)	
    local data = {}
    local targets = {}
    for i = 1,test_size do
        local observation, target = unpack(dataInfo.nextExample())
        table.insert(data, observation:clone())
        table.insert(targets, target:clone())
    end
    torch.save('th-mnist/test_'..num_dists..'.t7', {data, targets})
end
