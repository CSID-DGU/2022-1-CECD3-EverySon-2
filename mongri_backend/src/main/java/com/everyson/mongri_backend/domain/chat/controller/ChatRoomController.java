package com.everyson.mongri_backend.domain.chat.controller;

import com.everyson.mongri_backend.domain.chat.service.ChatRoomService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequiredArgsConstructor
@RequestMapping("/usr/chat")
public class ChatRoomController {

    private final ChatRoomService chatRoomService;

    @GetMapping("/room")
    public ResponseEntity getChatRoom() {
        Map<String, Object> map = new HashMap<String, Object>();

        map.put("roomID", chatRoomService.createRoom());
        map.put("message", "안녕하세요, 사용자님.\n오늘 하루는 어떠셨나요?");

        return ResponseEntity.ok(map);
    }
}
