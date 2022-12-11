package com.everyson.mongri_backend.domain.chat.controller;

import com.everyson.mongri_backend.domain.chat.service.ChatbotService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.util.*;

@RestController
@RequiredArgsConstructor
public class ChatMessageController {

    private final ChatbotService chatbotService;

    @PostMapping("/usr/chatbot")
    public ResponseEntity getChatbotRes(@RequestBody Map<String, Object> request) {
        return ResponseEntity.ok(chatbotService.generateChatbotRes(request.get("text")));
    }
}